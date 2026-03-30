import torch
import torch.nn.functional as F

import utils.config as cfg
from utils.utils import SoftmaxCategorical
from tqdm import tqdm
from model.model_components import InputProcessingImage
from collections import defaultdict

# --------------------------------------------------
# Sampler for Algorithm 1 (OA-ARDM sampling step)
# --------------------------------------------------
class Sampler_OA_ARDMs:
    def __init__(self, model, num_classes=256, absorbing_value=0, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.absorbing_value = absorbing_value

        self.input_processing = InputProcessingImage(
            num_classes=num_classes
        ).to(device)

    # --------------------------------------------------
    # Sample permutation σ
    # --------------------------------------------------
    def sample_sigma(self, B, D):
        return torch.stack([
            torch.randperm(D, device=self.device) for _ in range(B)
        ])

    # --------------------------------------------------
    # Build mask m = (σ < t)
    # --------------------------------------------------
    def build_mask(self, sigma, t, H, W):
        B, D = sigma.shape
        mask = (sigma < t.unsqueeze(1)).float()  # (B, D)
        return mask.view(B, 1, H, W)

        # for b in range(B):
        #     mask[b, sigma[b, :t-1]] = 1     # mask doesen't include this time

        # return mask.view(B, 1, H, W)

    # --------------------------------------------------
    # Sampling (Algorithm 1)
    # --------------------------------------------------
    @torch.no_grad()
    def sample(self, B, H, W):
        self.model.eval()

        D = H * W

        # ----------------------------------
        # Step 0: initialize x = a
        # ----------------------------------
        x = torch.full(
            (B, 1, H, W),
            fill_value=self.absorbing_value,
            dtype=torch.long,
            device=self.device
        )

        # ----------------------------------
        # Step 1: sample permutation σ
        # ----------------------------------
        sigma = self.sample_sigma(B, D)

        # flatten view for indexing
        x_flat = x.view(B, -1)

        # ----------------------------------
        # Step 2: iterative generation
        # ----------------------------------
        for t in range(D):
            # mask m = σ < t
            mask = self.build_mask(sigma, t, H, W)    # dont include this time on mask

            # masked input
            x_masked = x * mask + self.absorbing_value * (1 - mask)

            # forward
            features, temb = self.input_processing(x_masked, torch.full((B,), t, device=self.device), mask)
            logits = self.model(features, temb)  # (B, C, H, W)

            # reshape logits
            logits = logits.permute(0, 2, 3, 1).reshape(B, D, self.num_classes)

            # ----------------------------------
            # sample ONLY position σ(t)
            # ----------------------------------
            batch_idx = torch.arange(B)
            pixel_idx = sigma[:, t]
            probs = torch.softmax(logits[batch_idx, pixel_idx], dim=-1)
            samples = torch.multinomial(probs, 1).squeeze(-1)
            x_flat[batch_idx, pixel_idx] = samples

        return x

# --------------------------------------------------
# Trainer for Algorithm 2 (OA-ARDM optimizing step)
# --------------------------------------------------
class Trainer_OA_ARDMs:
    """
    Implements all steps of Algorithm 2 (training OA-ARDMs):

        Step 1: Sample timestep t ~ Uniform(1,...,D)
        Step 2: Sample a random permutation σ ~ Uniform(S_D)
        Step 3: Construct mask m = (σ < t)
        Step 4: Apply mask to input: i = m*x + (1-m)*a
        Step 5: Pass masked input through InputProcessingImage
        Step 6: Pass features through model to predict logits θ
        Step 7: Compute cross-entropy loss on masked (unobserved) pixels

    Inputs:
        x: (B,1,H,W) integer tensor with pixel values in [0, num_classes-1]

    Outputs:
        loss: scalar, average cross-entropy loss over batch
    """

    def __init__(self, model, num_classes=256, absorving_value=0, device="cuda"):
        # InputProcessingImage instance
        self.device=device
        self.input_processing = InputProcessingImage(
                                    num_classes=num_classes,
                                    max_time=num_classes        # max time = num pixels
                                    ).to(self.device)  
        self.model = model                  # Neural network predicting θ
        self.absorbing_value = absorving_value
        self.num_classes = num_classes

    # --------------------------------------------------
    # Step 1: Sample timestep t
    # --------------------------------------------------
    def sample_t(self, B, D, device):
        """
        Sample a timestep for each datapoint in the batch t ~ U(1,…,D).

        Inputs:
            B: batch size
            D: total number of pixels (H*W)
            device: torch.device

        Outputs:
            t: (B,) tensor with integers in [1, D]
        """
        return torch.randint(1, D + 1, (B,), device=device)

    # --------------------------------------------------
    # Step 2: Sample permutation σ
    # --------------------------------------------------
    def sample_sigma(self, B, D, device):
        """
        Sample a random permutation of pixels for each batch element σ~U(SD).

        Inputs:
            B: batch size
            D: total number of pixels
            device: torch.device

        Outputs:
            sigma: (B, D) tensor, each row is a permutation of pixel indices
        """
        return torch.stack([torch.randperm(D, device=device) for _ in range(B)])

    # --------------------------------------------------
    # Step 3: Build mask m = (σ < t)
    # --------------------------------------------------
    def build_mask(self, sigma, t, H, W):
        """
        Construct the binary mask indicating observed pixels i = m*x + (1-m)*a.

        Inputs:
            sigma: (B, D) permutations
            t: (B,) timestep
            H, W: image height and width

        Outputs:
            mask: (B, 1, H, W), 1=observed, 0=masked/unobserved
        """
        B, D = sigma.shape
        mask = (sigma < t.unsqueeze(1)).float()  # (B, D)
        return mask.view(B, 1, H, W)
        # for i in range(B):
        #     mask[i, sigma[i, :t[i]-1]] = 1  # first t[i] pixels in permutation are observed
        # return mask.view(B, 1, H, W)

    # --------------------------------------------------
    # Step 4: Apply mask to input
    # --------------------------------------------------
    def apply_mask(self, x, mask):
        """
        Apply absorbing state to unobserved pixels. (1 − m) 

        Inputs:
            x: (B,1,H,W) integer tensor
            mask: (B,1,H,W), 1=observed, 0=masked

        Outputs:
            x_masked: (B,1,H,W) tensor with masked pixels replaced by absorbing_value
        """
        x_masked = x * mask + self.absorbing_value * (1 - mask)
        return x_masked.long()

    # --------------------------------------------------
    # Full Algorithm 2 forward + loss
    # --------------------------------------------------
    def __call__(self, x, return_per_sample=False):

        """
        Run a full Algorithm 2 training step.

        Inputs:
            x: (B,1,H,W) integer tensor
            return_per_sample: Flag to give loss per digit and image 4 eval

        Outputs:
            loss: scalar, cross-entropy loss over masked pixels
        """
        B, _, H, W = x.shape
        D = H * W
        device = x.device

        # -----------------------------
        # Steps 1–4: sample t, σ, build mask, apply mask
        # -----------------------------
        t = self.sample_t(B, D, device)
        sigma = self.sample_sigma(B, D, device)
        mask = self.build_mask(sigma, t, H, W)  # dont include this time on mask
        x_masked = self.apply_mask(x, mask)

        # -----------------------------
        # Step 5: pass through InputProcessingImage
        # -----------------------------
        features, temb = self.input_processing(x_masked, t, mask)

        # Step 6: predict logits θ
        logits = self.model(features, temb)  # (B, num_classes, H, W)

        # Step 7: compute loss on masked pixels
        target = x.view(B, -1)  # flatten
        logits = logits.permute(0,2,3,1).reshape(B, -1, self.num_classes)  # (B,D,num_classes)
        mask_flat = mask.view(B, -1) == 0  # only masked pixels

        # Compute cross-entropy per batch, average aaply 1/D − t + 1 * Log
        losses = []

        for b in range(B):
            if mask_flat[b].sum() > 0:
                l = F.cross_entropy(
                    logits[b][mask_flat[b]],
                    target[b][mask_flat[b]],
                    reduction='mean'
                ) * D

                losses.append(l)
            else:
                losses.append(torch.tensor(0.0, device=device))

        losses = torch.stack(losses)

        if return_per_sample:
            return losses  # (B,)
        else:
            return losses.mean()

# ---------------------------
# High level trainer
# ---------------------------
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 num_classes=256, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dist = SoftmaxCategorical(n_channels=1, n_classes=2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.algorithm1 = Sampler_OA_ARDMs(self.model, num_classes=num_classes)
        self.algorithm2 = Trainer_OA_ARDMs(self.model, num_classes=num_classes)

    def train_step(self, x):
        imgs = x.to(self.device).long()  # MNIST binarized 0/1
        self.optimizer.zero_grad()
        loss = self.algorithm2(imgs)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def val_step(self, x, y, K=5):
        # Apply val step K times for avoid too much variance
        imgs = x.to(self.device).long()
        y = y.to(self.device)
        losses = torch.zeros(imgs.size(0), device=self.device)
        for _ in range(K):
            losses += self.algorithm2(imgs, return_per_sample=True)
        return losses/K, y

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        train_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for imgs, _ in train_bar:
            loss = self.train_step(imgs)
            total_loss += loss * imgs.size(0)
            train_bar.set_postfix({"loss": f"{loss:.4f}"})
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        digit_losses = defaultdict(list)    # capture loss per digit

        val_bar = tqdm(self.val_loader, desc="Validation", leave=False)
        for x, y in val_bar:
            x = x.to(self.device).long()
            y = y.to(self.device)
            losses, labels = self.val_step(x, y)    # losses (B,)
            for l, d in zip(losses, labels):
                digit_losses[int(d.item())].append(l.item())

        # average per digit
        digit_avg = { d: sum(v) / len(v) for d, v in digit_losses.items() }
        return digit_avg
    
    def fit(self, epochs):
        history = {
            "train_loss": [],
            "val_loss_per_digit": []
        }

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss = self.train_epoch()
            val_metrics = self.val_epoch()

            history["train_loss"].append(train_loss)
            history["val_loss_per_digit"].append(val_metrics)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss per digit: {val_metrics}")

        return history

    @torch.no_grad()
    def sample_images(self, n=3):
        self.model.eval()
        samples = self.algorithm1.sample(n, 28, 28)
        return samples.squeeze(1).cpu().numpy()


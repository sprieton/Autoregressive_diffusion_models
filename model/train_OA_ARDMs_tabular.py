"""OA-ARDM Algorithms 1 (sampling) and 2 (training) for tabular data.

Tabular sibling of train_OA_ARDMs.py. The logic is identical to the image
version but operates on (B, D) integer vectors and adds two tabular specifics:
    - a dedicated absorbing token = num_classes - 1 (no collision with data)
    - per-column logit masking: each column only scores its own categories.
"""

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from model.model_components_tabular import InputProcessingTabular


class Trainer_OA_ARDMs_Tabular:
    """Algorithm 2: sampled single-step ELBO term over a random permutation."""

    def __init__(self, model, num_classes, D, valid_mask, emb_dim=64, device="cuda"):
        self.device = device
        self.model = model.to(device)
        self.num_classes = num_classes
        self.D = D
        self.absorbing_value = num_classes - 1
        self.valid_mask = valid_mask.to(device)            # (D, num_classes)
        self.input_processing = InputProcessingTabular(
            num_classes=num_classes, D=D, emb_dim=emb_dim, max_time=D
        ).to(device)

    def sample_t(self, B):
        return torch.randint(1, self.D + 1, (B,), device=self.device)

    def sample_sigma(self, B):
        return torch.stack(
            [torch.randperm(self.D, device=self.device) for _ in range(B)]
        )

    def build_mask(self, sigma, t):
        # m = (sigma < t): 1 = observed, 0 = masked/unobserved
        return (sigma < t.unsqueeze(1)).float()

    def apply_mask(self, x, mask):
        return (x * mask + self.absorbing_value * (1 - mask)).long()

    def mask_invalid_logits(self, logits):
        # (B, D, C): force categories that don't exist in a column to -inf
        return logits.masked_fill(~self.valid_mask.unsqueeze(0), float("-inf"))

    def __call__(self, x, return_per_sample=False):
        B, D = x.shape
        t = self.sample_t(B)
        sigma = self.sample_sigma(B)
        mask = self.build_mask(sigma, t)
        x_masked = self.apply_mask(x, mask)

        feats, temb = self.input_processing(x_masked, t, mask)
        logits = self.mask_invalid_logits(self.model(feats, temb))   # (B, D, C)

        target = x.long()
        unobserved = mask == 0
        losses = []
        for b in range(B):
            sel = unobserved[b]
            if sel.any():
                l = F.cross_entropy(
                    logits[b][sel], target[b][sel], reduction="mean"
                ) * D
                losses.append(l)
            else:
                losses.append(torch.tensor(0.0, device=self.device))
        losses = torch.stack(losses)
        return losses if return_per_sample else losses.mean()


class Sampler_OA_ARDMs_Tabular:
    """Algorithm 1: generate rows one column at a time along a random order."""

    def __init__(self, model, input_processing, num_classes, D, valid_mask, device="cuda"):
        self.model = model
        self.input_processing = input_processing
        self.num_classes = num_classes
        self.D = D
        self.absorbing_value = num_classes - 1
        self.valid_mask = valid_mask.to(device)
        self.device = device

    @torch.no_grad()
    def sample(self, B):
        self.model.eval()
        x = torch.full(
            (B, self.D), self.absorbing_value, dtype=torch.long, device=self.device
        )
        sigma = torch.stack(
            [torch.randperm(self.D, device=self.device) for _ in range(B)]
        )
        # Reveal positions in value order: the mask marks j observed iff
        # sigma[j] < t, so the position generated at step t is argsort(sigma)[t].
        order = sigma.argsort(dim=1)
        idx = torch.arange(B, device=self.device)
        for t in range(self.D):
            mask = (sigma < t).float()
            feats, temb = self.input_processing(
                x, torch.full((B,), t, device=self.device), mask
            )
            logits = self.model(feats, temb)
            logits = logits.masked_fill(~self.valid_mask.unsqueeze(0), float("-inf"))
            pos = order[:, t]
            probs = torch.softmax(logits[idx, pos], dim=-1)
            x[idx, pos] = torch.multinomial(probs, 1).squeeze(-1)
        return x


class TabularTrainer:
    """High-level training/eval loop, mirroring the MNIST Trainer."""

    def __init__(self, model, train_loader, val_loader, num_classes, D,
                 valid_mask, target_idx, emb_dim=64, lr=1e-3, device="cuda"):
        self.device = device
        self.model = model.to(device)
        self.target_idx = target_idx
        self.D = D
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.algo2 = Trainer_OA_ARDMs_Tabular(
            model, num_classes, D, valid_mask, emb_dim, device
        )
        # The input-processing embeddings are learnable and must be optimized
        # alongside the network; the sampler reuses this same trained module.
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters())
            + list(self.algo2.input_processing.parameters()), lr=lr
        )
        self.sampler = Sampler_OA_ARDMs_Tabular(
            model, self.algo2.input_processing, num_classes, D, valid_mask, device
        )

    def train_epoch(self):
        self.model.train()
        total = 0.0
        for (x,) in tqdm(self.train_loader, desc="train", leave=False):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            loss = self.algo2(x)
            loss.backward()
            self.optimizer.step()
            total += loss.item() * x.size(0)
        return total / len(self.train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader, K=5):
        """Average NLL (nats) and bits-per-dimension, overall and per target class.

        The training loss already estimates the per-sample NLL in nats (the
        sampled term of the likelihood bound, scaled by D), so
        bpd = NLL_nats / (D * ln 2).
        """
        self.model.eval()
        per_class = defaultdict(list)
        all_nll = []
        for (x,) in tqdm(loader, desc="eval", leave=False):
            x = x.to(self.device)
            losses = torch.zeros(x.size(0), device=self.device)
            for _ in range(K):
                losses += self.algo2(x, return_per_sample=True)
            losses /= K
            all_nll.extend(l.item() for l in losses)
            for l, c in zip(losses, x[:, self.target_idx]):
                per_class[int(c.item())].append(l.item())

        denom = self.D * math.log(2.0)
        nll = sum(all_nll) / len(all_nll)
        per_class_nll = {c: sum(v) / len(v) for c, v in per_class.items()}
        return {
            "nll": nll,
            "bpd": nll / denom,
            "per_class_nll": per_class_nll,
            "per_class_bpd": {c: v / denom for c, v in per_class_nll.items()},
        }

    def fit(self, epochs, K=5):
        history = {"train_loss": [], "val_nll": [], "val_bpd": [],
                   "val_per_class_bpd": []}
        for e in range(epochs):
            tr = self.train_epoch()
            va = self.evaluate(self.val_loader, K)
            history["train_loss"].append(tr)
            history["val_nll"].append(va["nll"])
            history["val_bpd"].append(va["bpd"])
            history["val_per_class_bpd"].append(va["per_class_bpd"])
            print(f"Epoch {e + 1}/{epochs}  train NLL {tr:.3f}  "
                  f"val bpd {va['bpd']:.4f}  per-class bpd "
                  f"{ {c: round(v, 3) for c, v in va['per_class_bpd'].items()} }")
        return history

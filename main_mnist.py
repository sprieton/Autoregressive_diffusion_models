# main_mnist.py — OA-ARDM on binarized MNIST with train/val/test + bits-per-dim.
# Reuses the existing image Trainer / Algorithm 1 & 2 (model/train_OA_ARDMs.py);
# adds a proper held-out split, bpd reporting, per-digit eval and a sample grid.
import os
import math
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import utils.config as cfg
from model.models import LeNetWithTime, TinyTimeViT
from model.train_OA_ARDMs import Trainer

D = cfg.img_h * cfg.img_w           # 784
DENOM = D * math.log(2.0)
NUM_CLASSES = 2                     # binarized
EPOCHS = 15
VAL_K = 3


class Binarize:
    def __call__(self, t):
        return (t > 0.5).float()


@torch.no_grad()
def sample_grid(trainer, n, H=28, W=28):
    """Algorithm 1 sampling, fixing three issues in the committed Sampler:
    (1) it passed an int timestep to build_mask (expects a tensor);
    (2) it fed a float tensor to input_processing (asserts int dtype);
    (3) it used its OWN untrained InputProcessingImage and revealed the wrong
        position. We use the *trained* encoder (algorithm2.input_processing)
        and reveal positions in value order, consistent with build_mask."""
    model = trainer.model.eval()
    ip = trainer.algorithm2.input_processing      # the trained encoder
    absorbing = trainer.algorithm2.absorbing_value
    num_classes = trainer.algorithm2.num_classes
    device = trainer.device
    Dloc = H * W
    x = torch.full((n, 1, H, W), absorbing, dtype=torch.long, device=device)
    sigma = torch.stack([torch.randperm(Dloc, device=device) for _ in range(n)])
    order = sigma.argsort(dim=1)
    x_flat = x.view(n, -1)
    idx = torch.arange(n, device=device)
    for t in range(Dloc):
        t_tensor = torch.full((n,), t, device=device)
        mask = (sigma < t_tensor.unsqueeze(1)).float().view(n, 1, H, W)
        x_masked = (x * mask + absorbing * (1 - mask)).long()
        feats, temb = ip(x_masked, t_tensor, mask)
        logits = model(feats, temb)
        logits = logits.permute(0, 2, 3, 1).reshape(n, Dloc, num_classes)
        pos = order[:, t]
        probs = torch.softmax(logits[idx, pos], dim=-1)
        x_flat[idx, pos] = torch.multinomial(probs, 1).squeeze(-1)
    return x.squeeze(1).cpu().numpy()


@torch.no_grad()
def evaluate(trainer, loader, device, K=VAL_K):
    """Per-sample NLL (nats) -> overall and per-digit bits-per-dimension."""
    trainer.model.eval()
    from collections import defaultdict
    per_digit = defaultdict(list)
    all_nll = []
    for x, y in loader:
        x = x.to(device).long()
        losses = torch.zeros(x.size(0), device=device)
        for _ in range(K):
            losses += trainer.algorithm2(x, return_per_sample=True)
        losses /= K
        all_nll.extend(l.item() for l in losses)
        for l, d in zip(losses, y):
            per_digit[int(d)].append(l.item())
    nll = sum(all_nll) / len(all_nll)
    per_digit_bpd = {d: (sum(v) / len(v)) / DENOM for d, v in per_digit.items()}
    return {"nll": nll, "bpd": nll / DENOM, "per_digit_bpd": per_digit_bpd}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.results_dir, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor(), Binarize()])

    full_train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_set, val_set = random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=cfg.bach_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.bach_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.bach_size, shuffle=False)
    print(f"MNIST: D={D}, train={len(train_set)}, val={len(val_set)}, "
          f"test={len(test_set)}, device={device}")

    summary = []
    models = {"LeNetWithTime": LeNetWithTime(), "TinyTimeViT": TinyTimeViT()}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        trainer = Trainer(model, train_loader, val_loader, test_loader,
                          num_classes=NUM_CLASSES, device=device)
        # The committed Trainer optimizes only model.parameters(), leaving the
        # input-processing encoder at random init. Include it so the encoder is
        # actually trained (and reused at sampling time).
        trainer.optimizer = torch.optim.Adam(
            list(trainer.model.parameters())
            + list(trainer.algorithm2.input_processing.parameters()), lr=1e-3)
        rows = []
        for ep in range(EPOCHS):
            tr_nll = trainer.train_epoch()
            va = evaluate(trainer, val_loader, device)
            rows.append({"epoch": ep + 1, "train_nll": tr_nll,
                         "train_bpd": tr_nll / DENOM, "val_bpd": va["bpd"]})
            print(f"Epoch {ep + 1}/{EPOCHS}  train bpd {tr_nll / DENOM:.4f}  "
                  f"val bpd {va['bpd']:.4f}")

        te = evaluate(trainer, test_loader, device)
        samples = sample_grid(trainer, n=64)              # (64, 28, 28)
        np.save(f"{cfg.results_dir}/mnist_samples_{name}.npy", samples)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(rows).to_csv(
            f"{cfg.results_dir}/mnist_{name}_history_{ts}.csv", index=False)

        print(f"{name}: train bpd {rows[-1]['train_bpd']:.4f} | "
              f"val bpd {rows[-1]['val_bpd']:.4f} | test bpd {te['bpd']:.4f}")
        summary.append({"model": name, "train_bpd": rows[-1]["train_bpd"],
                        "val_bpd": rows[-1]["val_bpd"], "test_bpd": te["bpd"]})

    pd.DataFrame(summary).to_csv(f"{cfg.results_dir}/mnist_summary.csv", index=False)
    print("\n" + pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()

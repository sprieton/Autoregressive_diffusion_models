# make_figures.py — build the report figures from results/ artifacts.
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "./results"
FIG = "./figs"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


# ---------------------------------------------------------------- Fig 2: curves
def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    panels = [
        (axes[0], "MNIST", ["LeNetWithTime", "TinyTimeViT"], "mnist"),
        (axes[1], "Mushroom", ["MLPWithTime", "TabTransformer"], "tabular"),
    ]
    for ax, title, models, prefix in panels:
        for m in models:
            f = latest(f"{RES}/{prefix}_{m}_history_*.csv")
            if not f:
                continue
            df = pd.read_csv(f)
            line, = ax.plot(df["epoch"], df["train_bpd"], label=f"{m} train")
            ax.plot(df["epoch"], df["val_bpd"], "--", color=line.get_color(),
                    label=f"{m} val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("bits / dim")
        ax.legend(fontsize=6, frameon=False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/curves.pdf")
    print("wrote figs/curves.pdf")


# -------------------------------------------------------- Fig 3: MNIST samples
def fig_mnist_samples(model="LeNetWithTime"):
    path = f"{RES}/mnist_samples_{model}.npy"
    if not os.path.exists(path):
        print("no MNIST samples yet:", path)
        return
    s = np.load(path)[:64]
    fig, axes = plt.subplots(8, 8, figsize=(4.0, 4.0))
    for i, ax in enumerate(axes.flat):
        ax.imshow(s[i], cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(f"OA-ARDM samples ({model})", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/mnist_samples.pdf")
    print("wrote figs/mnist_samples.pdf")


# ----------------------------------------------- Fig 4: tabular marginal fidelity
def fig_tabular_marginals(models=("MLPWithTime", "TabTransformer")):
    """Real vs generated category marginals for both models, side by side
    (one row per model, three representative columns)."""
    meta = json.load(open(f"{RES}/tabular_meta.json"))
    cols, card = meta["columns"], meta["cardinalities"]
    real = np.load(f"{RES}/tabular_real_test.npy")

    tgt = meta["target_idx"]
    hi = int(np.argmax(card))                       # highest-cardinality column
    mid = int(np.argsort(card)[len(card) // 2])     # a mid-cardinality column
    chosen = list(dict.fromkeys([tgt, hi, mid]))[:3]

    fig, axes = plt.subplots(len(models), len(chosen), figsize=(7.0, 4.0))
    for r, model in enumerate(models):
        gen = np.load(f"{RES}/tabular_gen_{model}.npy")
        for c, j in enumerate(chosen):
            ax = axes[r, c]
            k = card[j]
            rp = np.bincount(real[:, j], minlength=k)[:k] / len(real)
            gp = np.bincount(gen[:, j], minlength=k)[:k] / len(gen)
            x = np.arange(k)
            ax.bar(x - 0.2, rp, width=0.4, label="real")
            ax.bar(x + 0.2, gp, width=0.4, label="generated")
            ax.set_xticks(x)
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f"{cols[j]} ({k})", fontsize=8)
            if c == 0:
                ax.set_ylabel(model, fontsize=8)
            if r == 0 and c == len(chosen) - 1:
                ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/tabular_marginals.pdf")
    print("wrote figs/tabular_marginals.pdf")


if __name__ == "__main__":
    fig_curves()
    fig_mnist_samples()
    fig_tabular_marginals()

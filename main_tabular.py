# main_tabular.py — OA-ARDM on the UCI Mushroom dataset (tabular categorical).
import os
from datetime import datetime

import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

import utils.config as cfg
from utils.tabular_data import load_mushroom, marginal_tvd
from model.models_tabular import MLPWithTime, TabTransformer
from model.train_OA_ARDMs_tabular import TabularTrainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.results_dir, exist_ok=True)

    data = load_mushroom(val_frac=0.2, seed=0)
    D, C = data["D"], data["num_classes"]
    print(f"Mushroom: D={D} columns, num_classes={C} (absorbing={C - 1}), "
          f"train={len(data['train_x'])}, val={len(data['val_x'])}, device={device}")

    train_loader = DataLoader(TensorDataset(data["train_x"]),
                              batch_size=cfg.tab_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(data["val_x"]),
                            batch_size=cfg.tab_batch_size, shuffle=False)

    models = {
        "MLPWithTime": MLPWithTime(D, cfg.tab_emb_dim, C, cfg.tab_hidden_dim),
        "TabTransformer": TabTransformer(D, cfg.tab_emb_dim, C,
                                         cfg.tab_num_heads, cfg.tab_num_layers),
    }

    for name, model in models.items():
        print(f"\n=== {name} ===")
        trainer = TabularTrainer(
            model, train_loader, val_loader, C, D,
            data["valid_mask"], data["target_idx"],
            emb_dim=cfg.tab_emb_dim, device=device,
        )
        history = trainer.fit(cfg.tab_num_epochs, K=cfg.tab_val_K)

        gen = trainer.sampler.sample(2000).cpu()
        tvd = marginal_tvd(data["val_x"], gen, data["cardinalities"])
        print(f"{name} mean marginal TVD (val vs generated): {tvd:.4f}")

        rows = []
        for ep, (tr, vc) in enumerate(
            zip(history["train_loss"], history["val_loss_per_class"]), 1
        ):
            row = {"epoch": ep, "train_loss": tr}
            for c, v in vc.items():
                row[f"val_loss_class_{c}"] = v
            rows.append(row)
        rows[-1]["marginal_tvd"] = tvd
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"{cfg.results_dir}/tabular_{name}_history_{ts}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()

# main_tabular.py — OA-ARDM on the UCI Mushroom dataset (tabular categorical).
import os
import json
import math
from datetime import datetime

import numpy as np
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

    data = load_mushroom(val_frac=0.15, test_frac=0.15, seed=0)
    D, C = data["D"], data["num_classes"]
    print(f"Mushroom: D={D} columns, num_classes={C} (absorbing={C - 1}), "
          f"train={len(data['train_x'])}, val={len(data['val_x'])}, "
          f"test={len(data['test_x'])}, device={device}")

    train_loader = DataLoader(TensorDataset(data["train_x"]),
                              batch_size=cfg.tab_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(data["val_x"]),
                            batch_size=cfg.tab_batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(data["test_x"]),
                             batch_size=cfg.tab_batch_size, shuffle=False)

    # Save real test data + cardinalities once, for the marginal-fidelity figure.
    np.save(f"{cfg.results_dir}/tabular_real_test.npy", data["test_x"].numpy())
    with open(f"{cfg.results_dir}/tabular_meta.json", "w") as f:
        json.dump({"cardinalities": data["cardinalities"], "D": D,
                   "num_classes": C, "target_idx": data["target_idx"],
                   "columns": data["columns"]}, f)

    denom = D * math.log(2.0)
    summary = []
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

        # Final held-out evaluation and sampling.
        train_eval = trainer.evaluate(train_loader, K=cfg.tab_val_K)
        test_eval = trainer.evaluate(test_loader, K=cfg.tab_val_K)
        gen = trainer.sampler.sample(2000).cpu()
        tvd = marginal_tvd(data["test_x"], gen, data["cardinalities"])
        np.save(f"{cfg.results_dir}/tabular_gen_{name}.npy", gen.numpy())

        print(f"{name}: train bpd {train_eval['bpd']:.4f} | "
              f"val bpd {history['val_bpd'][-1]:.4f} | "
              f"test bpd {test_eval['bpd']:.4f} | marginal TVD {tvd:.4f}")

        # Per-epoch history (train + val), train_loss in nats, plus bpd columns.
        rows = []
        for ep in range(len(history["train_loss"])):
            rows.append({
                "epoch": ep + 1,
                "train_nll": history["train_loss"][ep],
                "train_bpd": history["train_loss"][ep] / denom,
                "val_bpd": history["val_bpd"][ep],
            })
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(rows).to_csv(
            f"{cfg.results_dir}/tabular_{name}_history_{ts}.csv", index=False)

        summary.append({
            "model": name,
            "train_bpd": train_eval["bpd"],
            "val_bpd": history["val_bpd"][-1],
            "test_bpd": test_eval["bpd"],
            "test_bpd_edible": test_eval["per_class_bpd"].get(
                _class_code(data, "e"), float("nan")),
            "test_bpd_poison": test_eval["per_class_bpd"].get(
                _class_code(data, "p"), float("nan")),
            "marginal_tvd": tvd,
        })

    pd.DataFrame(summary).to_csv(f"{cfg.results_dir}/tabular_summary.csv", index=False)
    print("\nSummary written to results/tabular_summary.csv")
    print(pd.DataFrame(summary).to_string(index=False))


def _class_code(data, letter):
    # Target labels were encoded by sorted unique value: 'e' -> 0, 'p' -> 1.
    return {"e": 0, "p": 1}[letter]


if __name__ == "__main__":
    main()

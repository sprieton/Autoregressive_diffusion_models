"""Tabular data utilities for the OA-ARDM (UCI Mushroom).

The model is discrete and order-agnostic, so categorical tabular data fits
directly: each row becomes an integer vector x of shape (D,), where column j
takes values in {0, ..., cardinalities[j]-1}. A single shared vocabulary of
size num_classes = max(cardinalities) + 1 is used; the last index is a
dedicated absorbing token that is never a valid category in any column.
"""

import numpy as np
import torch


def load_mushroom(val_frac=0.2, seed=0):
    """Fetch and integer-encode the UCI Mushroom dataset.

    The target (edible/poisonous) is modelled as one more categorical column,
    so the model learns the full joint over features + label.

    Returns a dict with:
        train_x, val_x : LongTensor (N, D) integer category codes
        cardinalities  : list[int] length D, number of categories per column
        num_classes    : int = max(cardinalities) + 1 (last index = absorbing)
        valid_mask     : BoolTensor (D, num_classes), True where category valid
        target_idx     : int, index of the target column (for per-class eval)
        D              : int, number of columns
    """
    from ucimlrepo import fetch_ucirepo
    import pandas as pd

    data = fetch_ucirepo(id=73)
    X = data.data.features.copy()
    y = data.data.targets.copy()
    df = pd.concat([X, y], axis=1).fillna("missing")

    # Drop constant columns (e.g. 'veil-type') — they carry no information.
    df = df.loc[:, df.nunique() > 1]

    target_name = y.columns[0]
    target_idx = list(df.columns).index(target_name)

    codes = np.zeros(df.shape, dtype=np.int64)
    cardinalities = []
    for j, col in enumerate(df.columns):
        cats = sorted(df[col].unique())
        mapping = {c: i for i, c in enumerate(cats)}
        codes[:, j] = df[col].map(mapping).to_numpy()
        cardinalities.append(len(cats))

    D = codes.shape[1]
    num_classes = max(cardinalities) + 1  # +1 reserved absorbing token

    valid_mask = torch.zeros(D, num_classes, dtype=torch.bool)
    for j, k in enumerate(cardinalities):
        valid_mask[j, :k] = True

    from sklearn.model_selection import train_test_split
    train_codes, val_codes = train_test_split(
        codes, test_size=val_frac, random_state=seed, stratify=codes[:, target_idx]
    )

    return {
        "train_x": torch.from_numpy(train_codes),
        "val_x": torch.from_numpy(val_codes),
        "cardinalities": cardinalities,
        "num_classes": num_classes,
        "valid_mask": valid_mask,
        "target_idx": target_idx,
        "D": D,
    }


def marginal_tvd(real, gen, cardinalities):
    """Mean total-variation distance between per-column category marginals.

    A simple, interpretable sample-quality metric: 0 means the generated data
    reproduces every column's marginal exactly, 1 means maximal disagreement.
    """
    tvds = []
    for j, k in enumerate(cardinalities):
        rp = torch.bincount(real[:, j], minlength=k).float()
        gp = torch.bincount(gen[:, j], minlength=k).float()
        rp /= rp.sum()
        gp /= gp.sum()
        tvds.append(0.5 * (rp - gp).abs().sum().item())
    return sum(tvds) / len(tvds)

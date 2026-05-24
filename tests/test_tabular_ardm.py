"""Tests for the tabular OA-ARDM. Most use a tiny synthetic dataset (no
network); `test_load_mushroom` is the only one that fetches from UCI."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_components_tabular import InputProcessingTabular
from model.models_tabular import MLPWithTime, TabTransformer
from model.train_OA_ARDMs_tabular import (
    Trainer_OA_ARDMs_Tabular,
    Sampler_OA_ARDMs_Tabular,
)

DEVICE = "cpu"


@pytest.fixture
def synthetic():
    """3 columns with cardinalities 2,3,4 -> num_classes = 5 (absorbing = 4)."""
    cardinalities = [2, 3, 4]
    D = len(cardinalities)
    num_classes = max(cardinalities) + 1
    valid_mask = torch.zeros(D, num_classes, dtype=torch.bool)
    for j, k in enumerate(cardinalities):
        valid_mask[j, :k] = True
    torch.manual_seed(0)
    cols = [torch.randint(0, k, (64,)) for k in cardinalities]
    x = torch.stack(cols, dim=1)  # (64, 3)
    return dict(x=x, D=D, num_classes=num_classes,
               valid_mask=valid_mask, cardinalities=cardinalities)


def make_model(D, emb_dim, num_classes):
    return MLPWithTime(D, emb_dim, num_classes, hidden_dim=64)


# ---------------------------------------------------------------- data loader
def test_load_mushroom():
    from utils.tabular_data import load_mushroom
    d = load_mushroom()
    assert d["D"] == 22
    assert d["num_classes"] == 13
    assert d["num_classes"] == max(d["cardinalities"]) + 1
    for j, k in enumerate(d["cardinalities"]):
        assert int(d["train_x"][:, j].max()) < k          # within column range
        assert int(d["train_x"][:, j].max()) != d["num_classes"] - 1  # not absorbing
    assert d["valid_mask"].sum().item() == sum(d["cardinalities"])


# ---------------------------------------------------- input processing & models
def test_input_processing_shapes(synthetic):
    ip = InputProcessingTabular(synthetic["num_classes"], synthetic["D"], emb_dim=16)
    x = synthetic["x"]
    t = torch.randint(1, synthetic["D"] + 1, (x.size(0),))
    mask = (torch.rand_like(x.float()) > 0.5)
    h, temb = ip(x, t, mask)
    assert h.shape == (x.size(0), synthetic["D"], 16)
    assert temb.shape == (x.size(0), 16)


@pytest.mark.parametrize("ctor", [
    lambda D, E, C: MLPWithTime(D, E, C, hidden_dim=64),
    lambda D, E, C: TabTransformer(D, E, C, num_heads=2, num_layers=1),
])
def test_models_output_shape(synthetic, ctor):
    D, C = synthetic["D"], synthetic["num_classes"]
    model = ctor(D, 16, C)
    feats = torch.randn(8, D, 16)
    temb = torch.randn(8, 16)
    out = model(feats, temb)
    assert out.shape == (8, D, C)


# --------------------------------------------------------------- core mechanics
def test_build_mask_observed_count(synthetic):
    tr = Trainer_OA_ARDMs_Tabular(
        make_model(synthetic["D"], 16, synthetic["num_classes"]),
        synthetic["num_classes"], synthetic["D"], synthetic["valid_mask"],
        emb_dim=16, device=DEVICE)
    B = 32
    sigma = tr.sample_sigma(B)
    t = torch.randint(1, synthetic["D"] + 1, (B,))
    mask = tr.build_mask(sigma, t)
    assert torch.equal(mask.sum(dim=1).long(), t)          # exactly t observed


def test_apply_mask_absorbing(synthetic):
    tr = Trainer_OA_ARDMs_Tabular(
        make_model(synthetic["D"], 16, synthetic["num_classes"]),
        synthetic["num_classes"], synthetic["D"], synthetic["valid_mask"],
        emb_dim=16, device=DEVICE)
    x = synthetic["x"]
    mask = torch.zeros_like(x).float()
    mask[:, 0] = 1.0                                       # only column 0 observed
    xm = tr.apply_mask(x, mask)
    assert torch.equal(xm[:, 0], x[:, 0])                  # observed unchanged
    assert (xm[:, 1:] == tr.absorbing_value).all()         # rest absorbed


def test_mask_invalid_logits(synthetic):
    tr = Trainer_OA_ARDMs_Tabular(
        make_model(synthetic["D"], 16, synthetic["num_classes"]),
        synthetic["num_classes"], synthetic["D"], synthetic["valid_mask"],
        emb_dim=16, device=DEVICE)
    logits = torch.zeros(4, synthetic["D"], synthetic["num_classes"])
    masked = tr.mask_invalid_logits(logits)
    for j, k in enumerate(synthetic["cardinalities"]):
        assert torch.isfinite(masked[:, j, :k]).all()      # valid finite
        assert torch.isinf(masked[:, j, k:]).all()         # invalid -inf


def test_trainer_loss_backprop(synthetic):
    model = make_model(synthetic["D"], 16, synthetic["num_classes"])
    tr = Trainer_OA_ARDMs_Tabular(
        model, synthetic["num_classes"], synthetic["D"],
        synthetic["valid_mask"], emb_dim=16, device=DEVICE)
    loss = tr(synthetic["x"])
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads)


def test_sampler_valid_categories(synthetic):
    model = make_model(synthetic["D"], 16, synthetic["num_classes"])
    tr = Trainer_OA_ARDMs_Tabular(
        model, synthetic["num_classes"], synthetic["D"],
        synthetic["valid_mask"], emb_dim=16, device=DEVICE)
    sampler = Sampler_OA_ARDMs_Tabular(
        model, tr.input_processing, synthetic["num_classes"],
        synthetic["D"], synthetic["valid_mask"], device=DEVICE)
    out = sampler.sample(50)
    assert out.shape == (50, synthetic["D"])
    for j, k in enumerate(synthetic["cardinalities"]):
        assert int(out[:, j].max()) < k                    # never absorbing/invalid


def test_overfit_decreases(synthetic):
    torch.manual_seed(0)
    model = make_model(synthetic["D"], 16, synthetic["num_classes"])
    tr = Trainer_OA_ARDMs_Tabular(
        model, synthetic["num_classes"], synthetic["D"],
        synthetic["valid_mask"], emb_dim=16, device=DEVICE)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(tr.input_processing.parameters()), lr=1e-2)
    x = synthetic["x"][:16]

    @torch.no_grad()
    def avg_loss(n=50):
        # The objective is stochastic in (t, sigma); average to reduce variance.
        return sum(tr(x).item() for _ in range(n)) / n

    first = avg_loss()
    for _ in range(300):
        opt.zero_grad()
        loss = tr(x)
        loss.backward()
        opt.step()
    assert avg_loss() < first                              # learning happens

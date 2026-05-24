# Tabular OA-ARDM (Mushroom) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the existing image (MNIST) OA-ARDM to a 100%-categorical tabular dataset (UCI Mushroom), as a parallel module that leaves the colleague's MNIST code untouched.

**Architecture:** Approach A — add tabular siblings to the existing modules. The OA-ARDM core (Algorithms 1 & 2) is re-expressed over `(B, D)` integer vectors instead of `(B,1,H,W)` images. Per-column heterogeneous cardinality is handled with a single shared vocabulary of size `num_classes = max_cardinality + 1`, where the last index is a dedicated absorbing token, and invalid category logits are masked to `-inf` per column.

**Tech Stack:** Python 3.10 (conda env `cv_lab`), PyTorch 2.10 (CUDA), pandas, scikit-learn, ucimlrepo, tqdm, pytest.

**Run everything with:** `/home/jorge/miniforge3/envs/cv_lab/bin/python`

---

## Design facts (probed from data)

- Dataset: UCI Mushroom (`fetch_ucirepo(id=73)`), 8124 rows, all-categorical.
- Modeled columns: 22 features + 1 target = 23, minus the constant `veil-type` column = **D = 22**.
- Target (`poisonous`, e/p) is modeled as one more categorical column → enables conditional + joint analysis.
- One feature (`stalk-root`) has missing values → filled as a `"missing"` category.
- Cardinalities range 2..12 → **num_classes = 13**, absorbing token index = 12 (never a valid category, so masked out everywhere in `valid_mask`).
- Per-target-class validation loss is the tabular analogue of the MNIST "loss per digit".

## File structure

- Create `utils/tabular_data.py` — load/encode Mushroom; build `valid_mask`; `marginal_tvd` metric.
- Create `model/model_components_tabular.py` — `InputProcessingTabular` (value+column+mask embeddings, time embedding).
- Create `model/models_tabular.py` — `MLPWithTime`, `TabTransformer` (mirror the MNIST LeNet/ViT pair).
- Create `model/train_OA_ARDMs_tabular.py` — `Trainer_OA_ARDMs_Tabular`, `Sampler_OA_ARDMs_Tabular`, `TabularTrainer`.
- Create `main_tabular.py` — entry point: load → train both models → sample → marginal TVD → CSV.
- Modify `utils/config.py` — append a TABULAR hyperparameter section (surgical).
- Create `tests/test_tabular_ardm.py` — pytest suite (synthetic fixtures + one network-backed loader test).
- Create `requirements.txt` — reproducibility for the repo/report.

---

## Task 0: Environment

- [ ] Install missing deps into `cv_lab`:
```bash
/home/jorge/miniforge3/envs/cv_lab/bin/python -m pip install -q tqdm ucimlrepo pytest
```
- [ ] Write `requirements.txt`:
```
torch
torchvision
pandas
scikit-learn
ucimlrepo
tqdm
matplotlib
```

## Task 1: Data loader (`utils/tabular_data.py`)

**Files:** Create `utils/tabular_data.py`; Test `tests/test_tabular_ardm.py`.

- [ ] Implement `load_mushroom(val_frac=0.2, seed=0)` returning a dict with `train_x, val_x` (LongTensor `(N,D)`), `cardinalities` (list len D), `num_classes` (= max card + 1), `valid_mask` (Bool `(D, num_classes)`), `target_idx` (int), `D` (int). Steps inside: fetch id=73, concat features+target, `fillna("missing")`, drop columns where `nunique()==1`, integer-encode each column by sorted unique values, build `valid_mask[j,:k]=True`, stratified `train_test_split` on the target column.
- [ ] Implement `marginal_tvd(real, gen, cardinalities)` → mean per-column total-variation distance between category marginals.
- [ ] Test `test_load_mushroom`: `D==22`, `num_classes==13`, every `train_x[:,j] < cardinalities[j]`, no value equals absorbing token, `valid_mask` has exactly `sum(cardinalities)` True entries. (network-backed)
- [ ] Run: `python -m pytest tests/test_tabular_ardm.py -k load -v` → PASS.

## Task 2: Input processing (`model/model_components_tabular.py`)

**Files:** Create `model/model_components_tabular.py`.

- [ ] Implement `InputProcessingTabular(num_classes, D, emb_dim=64, max_time=1000)`: `value_emb=Embedding(num_classes,E)`, `col_emb=Embedding(D,E)`, `mask_emb=Embedding(2,E)`, `time_embedding=TimeEmbedding(E, max_time)` (reuse `utils/utils.py`). `forward(x,t,mask)` returns `h=(B,D,E)` (sum of value+col+mask embeddings) and `temb=(B,E)`. Assert integer dtype.
- [ ] Test `test_input_processing_shapes` (synthetic): output shapes `(B,D,E)` and `(B,E)`.

## Task 3: Models (`model/models_tabular.py`)

**Files:** Create `model/models_tabular.py`.

- [ ] `MLPWithTime(D, emb_dim, num_classes, hidden_dim=512)`: flatten feats `(B,D*E)`, concat projected `temb`, 2 hidden ReLU layers, output `(B,D,num_classes)`.
- [ ] `TabTransformer(D, emb_dim, num_classes, num_heads=4, num_layers=2)`: add projected `temb` to each column token, `TransformerEncoder(batch_first=True)`, linear head → `(B,D,num_classes)`.
- [ ] Test `test_models_output_shape` (synthetic, both models): output `(B,D,num_classes)`.

## Task 4: Core algorithms (`model/train_OA_ARDMs_tabular.py`)

**Files:** Create `model/train_OA_ARDMs_tabular.py`.

- [ ] `Trainer_OA_ARDMs_Tabular(model, num_classes, D, valid_mask, emb_dim=64, device)`: holds its own `InputProcessingTabular(max_time=D)`; `absorbing_value=num_classes-1`. Methods: `sample_t` (`U[1,D]`), `sample_sigma` (random perms), `build_mask` (`σ<t`, 1=observed), `apply_mask` (masked→absorbing), `mask_invalid_logits` (`masked_fill(~valid_mask, -inf)`). `__call__(x, return_per_sample=False)`: forward, mask invalid logits, per-sample cross-entropy on unobserved positions × D; mean or per-sample.
- [ ] `Sampler_OA_ARDMs_Tabular(model, input_processing, num_classes, D, valid_mask, device)`: `sample(B)` initializes all-absorbing `(B,D)`, samples σ, iterates `t=0..D-1` filling position `σ[:,t]` from masked softmax. Returns `(B,D)` long.
- [ ] `TabularTrainer(...)`: Adam, wraps Trainer+Sampler. `train_epoch`, `val_epoch(K=5)` (per-target-class avg loss), `fit(epochs,K)`.
- [ ] Test `test_build_mask_observed_count` (synthetic): row with timestep t has exactly t observed entries.
- [ ] Test `test_apply_mask_absorbing`: masked positions become absorbing token, observed unchanged.
- [ ] Test `test_mask_invalid_logits`: invalid category logits are `-inf`, valid ones finite.
- [ ] Test `test_trainer_loss_backprop` (synthetic): loss is finite scalar; `.backward()` populates grads.
- [ ] Test `test_sampler_valid_categories` (synthetic): all sampled values `< cardinalities[j]` (never absorbing/invalid).
- [ ] Test `test_overfit_decreases` (synthetic, ~200 steps on a tiny repeated batch): final loss < initial loss.
- [ ] Run: `python -m pytest tests/test_tabular_ardm.py -v` → all PASS.

## Task 5: Config (`utils/config.py`)

**Files:** Modify `utils/config.py` (append only).

- [ ] Append TABULAR section: `tab_batch_size=256`, `tab_num_epochs=40`, `tab_emb_dim=64`, `tab_hidden_dim=512`, `tab_num_heads=4`, `tab_num_layers=2`, `tab_val_K=5`.

## Task 6: Entry point (`main_tabular.py`) + run

**Files:** Create `main_tabular.py`.

- [ ] Load data, build both models, train each, sample 2000, compute marginal TVD vs val set, save per-model CSV to `results/`. `os.makedirs(results_dir, exist_ok=True)`.
- [ ] Run: `python main_tabular.py` → both models train, NLL decreases, TVD printed and reasonable (< ~0.1), CSVs written.

## Task 7: Commit

- [ ] `git checkout -b tabular-oa-ardm`, add new files + config + plan/spec, commit.

---

## Self-review notes
- Spec coverage: tabular adaptation (Tasks 1-6), MNIST untouched (no edits to existing model files), report deferred per user. ✓
- No placeholders: all modules have concrete signatures and behavior. ✓
- Type consistency: `valid_mask` is `(D, num_classes)` everywhere; models return `(B,D,num_classes)`; data uses LongTensor `(N,D)`. ✓

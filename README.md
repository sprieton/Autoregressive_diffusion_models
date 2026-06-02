# OA-ARDM: Order-Agnostic Autoregressive Diffusion Models

A PyTorch implementation of Order-Agnostic Autoregressive Diffusion Models (OA-ARDMs) applied to two distinct data modalities: binarized images (**MNIST**) and fully categorical tabular data (**UCI Mushroom**).

**Authors:** Jorge Barcenilla and Santiago Prieto.

---

## 📖 Project Overview

This project **implements the Order-Agnostic Autoregressive Diffusion Model**, originally introduced by E. Hoogeboom et al. in "Autoregressive Diffusion Models" ([ICLR, 2022](https://arxiv.org/pdf/2110.02037)). The **OA-ARDM framework unifies deep autoregressive models**—which achieve exact likelihoods but impose a single fixed generation order—with discrete diffusion models, which destroy data toward an absorbing state and learn the reverse process. **This unification is achieved through a single shared network trained to predict an arbitrary masked** subset of variables, maximizing a one-term estimator of the likelihood bound ($\log p(x) \ge \mathbb{E}_{t}[D \cdot L_t]$). This yields a flexible generation order and enables adaptive parallel sampling.

While the original formulation focused primarily on homogeneous modalities with regular spatial or sequential structures (like images and text), this repository demonstrates that the same unmodified training and sampling algorithms generalize to heterogeneous categorical data.

### Modality Adaptations
The core training objective and sampling mechanisms remain identical across both datasets; only the input parameterization is adapted for each modality:
*   **Images (Binarized MNIST):** Utilizes a standard uniform structure ($28\times28$) where every variable shares a single vocabulary ($K=2$) and uses 0 as the absorbing value.
*   **Tabular Data (UCI Mushroom):** Consists of 22 categorical columns with differing cardinalities (ranging from 2 to 12), which breaks the single-vocabulary assumption. To solve this, the model uses one shared vocabulary with a dedicated absorbing token located at the last index (avoiding collisions with valid categories). Per-column validity is strictly enforced by masking the logits of absent categories to $-\infty$.

### Backbones and Evaluation
We evaluate two network families per modality:
*   **Images:** A convolutional network with time embedding (`LeNetWithTime`) and a small vision transformer (`TinyTimeViT`).
*   **Tabular:** A multilayer perceptron (`MLPWithTime`) and a Transformer encoder over column tokens (`TabTransformer`).

Models are evaluated using held-out negative log-likelihood measured in bits per dimension (bpd). For tabular data, sample fidelity is further quantified using the mean per-column total-variation distance (TVD) between the generated data and the empirical test set marginals.

---

## 📂 Repository Layout

* `model/`: Contains the shared algorithmic core for sampling (Algorithm 1) and training (Algorithm 2). It also houses the specific input parameterization architectures for images (`model_components.py`, `models.py`) and tabular data (`model_components_tabular.py`, `models_tabular.py`).
* `utils/`: Hyperparameter configurations (`config.py`), sinusoidal time embeddings, and tabular data handlers, including the Mushroom loader and the marginal-TVD metric logic.
* `tests/`: Unit tests validating the tabular pipeline's core mechanics (masking, absorbing tokens, loss back-propagation, and sampler validity).
* `main_mnist.py`: Main execution script to train and evaluate vision backbones on binarized MNIST, featuring bpd reporting and sample grid generation.
* `main_tabular.py`: Main execution script to train and evaluate tabular backbones on the UCI Mushroom dataset.
* `make_figures.py`: Post-processing script to generate likelihood curves, digit sample grids, and tabular marginal distribution bar charts from training artifacts.
* `main.py`: Original minimal runner for MNIST.

---

## ⚙️ Setup & Installation

This project requires PyTorch (CUDA support is optional but recommended). The MNIST dataset downloads automatically on its first run, and the Mushroom dataset is fetched dynamically via `ucimlrepo`.

To install the required dependencies, run:

```bash
pip install -r requirements.txt

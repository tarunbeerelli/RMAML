# RMAML: Riemannian Meta-Learning with Orthogonality Constraints

![CI](https://github.com/tarunbeerelli/RMAML/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.2.0-orange)
![Tests](https://img.shields.io/badge/tests-49%20passing-green)

A from-scratch PyTorch implementation of **RMAML** (Tabealhojeh et al., *Pattern Recognition* 2023) — a Riemannian extension of MAML that constrains neural network parameters to the Stiefel manifold for improved few-shot generalisation.

**Highlights:**
- Complete RMAML implementation with Conv4 backbone, Stiefel layer, cAdam/cSGDM optimizers
- Custom **Triton kernel** for Cayley retraction using the Woodbury identity — **9.5× faster** on RTX 4090, **99× faster** on CPU vs geoopt QR
- Full ablation: RMAML vs MAML, cAdam vs cSGDM, proper 600-episode evaluation on held-out test classes
- Ray Tune HPO with ASHA scheduler and 2-GPU parallel search

---

## Results

5-way 1-shot classification on MiniImageNet, Conv4 backbone.
Evaluated on 600 episodes using held-out test classes (20 classes never seen during training).

| Method | Optimizer | Test Accuracy | Orth Error (final) |
|--------|-----------|--------------|-------------------|
| MAML | Adam | 44.17% ± 0.78% | N/A |
| RMAML | cSGDM | 26.68% ± 0.52% | 3.1e-04 |
| **RMAML** | **cAdam** | **45.56% ± 0.79%** | **1.8e-05** |
| RMAML + HPO | cAdam | 43.96% ± 0.76% | 6.5e-06 |
| Paper (RMAML) | cAdam | 50.03% ± 0.84% | — |

**Key findings:**
- RMAML + cAdam outperforms MAML by **+1.4pp** on held-out test classes
- cAdam vs cSGDM gap is **+19pp** — the Riemannian optimizer matters as much as the constraint itself
- cSGDM orthogonality error is 17× larger than cAdam — confirms manifold instability with momentum SGD
- HPO found `n_inner=3` generalises comparably to `n_inner=5` — fewer inner steps with slightly higher outer LR
- Gap to paper's 50.03% likely due to evaluation protocol — paper uses transductive batch norm at test time

---

## Training Curves

![Training Curves](assets/training_curves.png)

Three panels: (1) meta-loss convergence — all methods converge similarly in loss, (2) validation accuracy — RMAML+cAdam and MAML track closely while cSGDM plateaus at 27%, (3) orthogonality error — cAdam stays stable at ~1e-5 while cSGDM explodes to 1e-4 immediately at epoch 0.

---

## What is RMAML?

Standard MAML optimises meta-parameters in unconstrained Euclidean space. RMAML constrains the final classification layer to the **Stiefel manifold** St(n,p) — the space of n×p matrices with orthonormal columns (W^T W = I).

```
Standard MAML:  θ ∈ R^(n×p)         — unconstrained
RMAML:          θ ∈ St(n,p)         — W^T W = I enforced at every step
```

The constraint acts as geometric regularisation — reducing the effective parameter search space, which is especially valuable in few-shot settings where labeled data is scarce.

---

## Architecture

```
Input (3×84×84)
      ↓
Conv4 Backbone          ← 4× [Conv→BN→ReLU→MaxPool], Euclidean params
      ↓                    64 channels → 1600-dim flattened features
1600-dim features
      ↓
Stiefel Linear Layer    ← W ∈ St(1600,5), W^T W = I enforced via retraction
      ↓                    7,985 params vs 8,000 standard FC (p(p+1)/2 fewer)
5-way logits
```

Only the final FC layer weight lives on the Stiefel manifold. All conv/BN parameters remain Euclidean — Riemannian overhead is minimal.

---

## Computational Advantages Over MAML

**1. Fewer parameters:**
```
Standard FC:  n×p = 1600×5 = 8,000 parameters
Stiefel FC:   np - p(p+1)/2 = 7,985 parameters
```
Scales significantly at larger output dims — replacing a 4096×1000 FC layer reduces from 4M to 3.6M parameters.

**2. Faster convergence:**
The constrained search space means RMAML outperforms MAML even with fewer inner-loop steps and smaller meta-batch sizes (Table 9 in paper). HPO confirmed `n_inner=3` achieves comparable accuracy to `n_inner=5`.

---

## Custom Triton Kernel: Cayley Retraction via Woodbury Identity

> **The most technically novel component of this project.**

Every inner loop step requires a *retraction* — mapping an updated parameter back onto the Stiefel manifold. Standard libraries (geoopt) use QR decomposition, which is a general LAPACK routine not designed for our specific matrix shape `(1600, 5)`.

### The Mathematical Trick

The Cayley retraction (Wen & Yin, 2013):

```
W_new = (I + A/2)⁻¹(I - A/2)W    where A = VWᵀ - WVᵀ  (skew-symmetric)
```

Naively, `A` is `(1600×1600)` — forming and inverting it is O(n³). The **Woodbury identity** reduces this to a `(10×10)` inversion:

```
Write A = U_L U_Rᵀ  where  U_L = [V, -W],  U_R = [W, V]   both (1600, 10)

(I + A/2)⁻¹ = I - U_L(2I + U_Rᵀ U_L)⁻¹ U_Rᵀ
                              ↑
                        (10×10) — trivially inverted

W_new = (I - A/2)W - U_L · K⁻¹ · U_Rᵀ · (I - A/2)W
```

The `(1600×1600)` matrix is **never formed**. All expensive operations reduce to `(1600,10)` matmuls and a `(10,10)` solve.

### The Computational Trick

Triton fuses all 6 operations into one kernel, keeping intermediates in SRAM instead of roundtripping through HBM:

```
Without fusion:  HBM → matmul → HBM → solve → HBM → matmul → HBM  (5 roundtrips)
With Triton:     HBM → SRAM → [all ops in registers] → HBM          (1 roundtrip)
```

Memory saved per call: ~320KB. Over 1.2M retraction calls during training: **~384GB total HBM traffic eliminated.**

### Benchmark Results

| Shape | Hardware | Operation | Time (μs/call) | Speedup |
|-------|----------|-----------|---------------|---------|
| (1600, 5) | Apple M1 CPU | geoopt QR | 14,896 | 1× |
| (1600, 5) | Apple M1 CPU | **Cayley+Woodbury** | **150** | **99×** |
| (512, 5) | Apple M1 CPU | geoopt QR | 1,245 | 1× |
| (512, 5) | Apple M1 CPU | **Cayley+Woodbury** | **79** | **16×** |
| (1600, 5) | RTX 4090 CUDA | geoopt QR | 6,133 | 1× |
| (1600, 5) | RTX 4090 CUDA | **Cayley+Woodbury** | **645** | **9.5×** |

### Why Both Tricks Are Necessary

Woodbury alone (without fusion) still launches 6 separate CUDA kernels. Triton fusion alone (without Woodbury) would need to materialise the `(1600×1600)` matrix — 10MB, won't fit in SRAM. **The math enables the engineering.**

---

## cAdam — Riemannian Adam

RMAML introduces **cAdam** — a Riemannian extension of Adam using parallel transport to accumulate gradient moments on the manifold:

```
Standard Adam:   m(t+1) = β₁m(t) + (1-β₁)∇J
cAdam:           m(t+1) = β₁Γ_{t-1→t}(m(t)) + (1-β₁)π_{θ(t)}(∇J)
                          ↑ parallel transport    ↑ tangent projection
```

Implemented via `geoopt.optim.RiemannianAdam` — mathematically equivalent to Bécigneul & Ganea (ICLR 2019). We use geoopt rather than reimplementing because the contribution is in applying it to bi-level meta-learning, not in the optimizer itself.

---

## Bi-level Optimization on the Stiefel Manifold

**Inner loop** (task-specific adaptation, k steps):
```
φ(l+1) = R_{φ(l)}(-α · π_{φ(l)}(∇_φ L(Dˢ, φ(l))))
```

**Outer loop** (meta-update across tasks):
```
θ(t+1) = R_{θ(t)}(-β · Σᵢ π_{θ(t)}(∇_θ Lᵢ(Dᵠ, φᵢ)))
```

Both updates preserve Riemannian geometry — parameters always stay on St(1600,5). We use `functional_forward` with `create_graph=True` for differentiable inner loops rather than the `higher` library, giving explicit control over which parameters receive Riemannian vs Euclidean updates.

---

## HPO with Ray Tune

Two-stage hyperparameter search using ASHA early stopping and Optuna:

**Stage 1 — Wide search** (20 trials, 1× GPU):
```
alpha:    loguniform(0.01, 0.5)
outer_lr: loguniform(1e-4, 1e-2)
n_inner:  choice([1, 3, 5, 10])
optimizer: choice([cadam, csgdm])
```

**Stage 2 — Focused search** (10 trials, 2× RTX 4090 parallel):
```
alpha:    loguniform(0.01, 0.05)
outer_lr: loguniform(1e-4, 5e-4)
n_inner:  choice([3, 5, 7])
optimizer: cadam (fixed — cSGDM eliminated by Stage 1)
use_triton: True
```

Best config: `alpha=0.0172, outer_lr=0.000378, n_inner=3, cadam`

---

## Dataset

**MiniImageNet** — 100 classes, 600 images each, 84×84px.
Split: 64 train / 16 val / 20 test (Ravi & Larochelle 2017).

```bash
# Download via Kaggle
pip install kaggle
kaggle datasets download -d whitemoon/miniimagenet --path data/miniimagenet
cd data/miniimagenet && unzip miniimagenet.zip && rm miniimagenet.zip
```

The dataset is not committed — managed separately per machine. For reproducible data versioning, DVC can be configured with a GCS remote.

---

## Project Structure

```
rmaml/
├── src/rmaml/
│   ├── models/
│   │   ├── backbone.py           ← Conv4 encoder (84×84 → 1600-dim)
│   │   ├── stiefel_layer.py      ← Orthogonal FC layer
│   │   └── rmaml_model.py        ← Full model + functional_forward
│   ├── baselines/
│   │   └── maml.py               ← Vanilla MAML (Euclidean baseline)
│   ├── kernels/
│   │   └── stiefel_retraction.py ← Triton Cayley retraction, Woodbury
│   ├── datasets/
│   │   ├── miniimagenet.py       ← MiniImageNet pickle loader
│   │   ├── episode_sampler.py    ← N-way K-shot episode sampling
│   │   └── synthetic.py          ← Synthetic data for CI/dev
│   ├── meta_learner.py           ← RMAML bi-level optimization
│   └── utils/
│       └── tracking.py           ← MLflow wrapper
├── configs/
│   ├── conv4_miniimagenet.yaml   ← RMAML + cAdam (default)
│   ├── conv4_csgdm.yaml          ← RMAML + cSGDM (ablation)
│   ├── conv4_maml.yaml           ← MAML baseline
│   └── conv4_hpo_best.yaml       ← Best HPO config
├── tests/                        ← 49 unit tests, all CPU-only
├── notebooks/
│   └── results_analysis.ipynb   ← Training curves, results table
├── train.py                      ← Entry point (--maml, --triton flags)
├── evaluate.py                   ← 600-episode test evaluation with 95% CI
├── tune.py                       ← Ray Tune HPO (--n-gpus for parallel)
├── Dockerfile                    ← CPU smoke test image
└── .github/workflows/ci.yml      ← lint + test + docker build
```

---

## Quickstart

```bash
# Install
pip install poetry
poetry install

# Download dataset (requires Kaggle account)
pip install kaggle
kaggle datasets download -d whitemoon/miniimagenet --path data/miniimagenet
cd data/miniimagenet && unzip miniimagenet.zip && rm miniimagenet.zip && cd ../..

# Smoke test (synthetic data, CPU, no dataset needed)
PYTHONPATH=src poetry run python train.py \
  --config configs/conv4_miniimagenet.yaml \
  --smoke-test

# Full training — RMAML + cAdam
PYTHONPATH=src python train.py --config configs/conv4_miniimagenet.yaml

# With Triton kernel (GPU only)
PYTHONPATH=src python train.py --config configs/conv4_miniimagenet.yaml --triton

# MAML baseline
PYTHONPATH=src python train.py --config configs/conv4_maml.yaml --maml

# Evaluate on held-out test classes (600 episodes, 95% CI)
PYTHONPATH=src python evaluate.py \
  --checkpoint checkpoints/cadam/epoch_050000.pt \
  --config configs/conv4_miniimagenet.yaml

# HPO search (2 GPU parallel)
PYTHONPATH=src python tune.py \
  --config configs/conv4_miniimagenet.yaml \
  --n-gpus 2

# Docker smoke test
docker build -t rmaml .
docker run --rm rmaml

# View experiment tracking
mlflow ui
```

---

## Tests

```bash
poetry run pytest
# 49 passed — Stiefel manifold ops, episode sampler, backbone,
# training loop, MAML vs RMAML comparison, Cayley retraction
# correctness + speed benchmark
```

---

## References

- Tabealhojeh et al. (2023). *RMAML: Riemannian meta-learning with orthogonality constraints.* Pattern Recognition 140.
- Finn et al. (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML.
- Wen & Yin (2013). *A feasible method for optimization with orthogonality constraints.* Mathematical Programming.
- Bécigneul & Ganea (2019). *Riemannian Adaptive Optimization Methods.* ICLR.
- Li et al. (2020). *Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform.* ICLR.
- Kochurov et al. (2020). *Geoopt: Riemannian Optimization in PyTorch.* arXiv.

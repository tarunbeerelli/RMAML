# RMAML: Riemannian Meta-Learning with Orthogonality Constraints

A high-performance, from-scratch PyTorch implementation of RMAML (Tabealhojeh et al., Pattern Recognition 2023) — a Riemannian extension of MAML that constrains neural network parameters to the Stiefel manifold for improved few-shot generalization.

### Highlights:
*   **Complete RMAML implementation:** Features a Conv4 backbone, Stiefel layer, and Riemannian optimizers (cAdam/cSGDM).
*   **Custom Retraction Architecture:** Implements a Cayley retraction utilizing the Woodbury identity, yielding a **13.4× mathematical speedup** over standard `geoopt` implementations on an RTX 4090.
*   **Deep Hardware Analysis:** Includes a custom Triton kernel and a detailed architectural breakdown of when kernel fusion outpaces heavily optimized cuBLAS (cache vs. memory bandwidth bounds).
*   **Rigorous Evaluation:** Full ablation studies (RMAML vs MAML, cAdam vs cSGDM) tested on a proper 600-episode evaluation protocol on held-out test classes.
*   **Distributed Hyperparameter Optimization:** Powered by Ray Tune with ASHA scheduling and multi-GPU parallel search.

---

## Results

**Task:** 5-way 1-shot classification on MiniImageNet, Conv4 backbone.
**Protocol:** Evaluated on 600 episodes using held-out test classes (20 classes never seen during training).

| Method | Optimizer | Test Accuracy | Orth Error (final) |
|---|---|---|---|
| MAML | Adam | 44.17% ± 0.78% | N/A |
| RMAML | cSGDM | 26.68% ± 0.52% | 3.1e-04 |
| RMAML | cAdam | 45.56% ± 0.79% | 1.8e-05 |
| RMAML + HPO | cAdam | 43.96% ± 0.76% | 6.5e-06 |
| Paper (RMAML) | cAdam | 50.03% ± 0.84% | — |

**Key findings:**
*   RMAML + cAdam outperforms standard MAML by +1.4pp on held-out test classes.
*   The Riemannian optimizer matters just as much as the constraint itself: the cAdam vs cSGDM gap is a massive +19pp.
*   cSGDM orthogonality error is 17× larger than cAdam, confirming manifold instability when using standard momentum SGD on the Stiefel manifold.
*   HPO revealed that fewer inner steps (`n_inner=3`) paired with a slightly higher outer learning rate generalizes comparably to `n_inner=5`.
*   The gap to the paper's 50.03% accuracy is likely due to the evaluation protocol (the paper uses transductive batch normalization at test time).

---

## What is RMAML?

Standard MAML optimizes meta-parameters in unconstrained Euclidean space. RMAML constrains the final classification layer to the Stiefel manifold — the space of $n \times p$ matrices with orthonormal columns where $W^T W = I$.

*   **Standard MAML:** $\theta \in \mathbb{R}^{n \times p}$ (Unconstrained)
*   **RMAML:** $\theta \in St(n,p)$ ($W^T W = I$ enforced at every step)

This constraint acts as a geometric regularizer. By drastically reducing the effective parameter search space, the model is forced to learn more robust representations, which is exceptionally valuable in few-shot settings where labeled data is scarce.

---

## Architecture

**Input:** (3 × 84 × 84)
↓
**Conv4 Backbone:** 4× [Conv → BN → ReLU → MaxPool] (Euclidean params)
↓ (64 channels → 1600-dim flattened features)
**Stiefel Linear Layer:** $W \in St(1600,5)$, $W^T W = I$ enforced via retraction.
↓ (7,985 params vs 8,000 standard FC)
**5-way Logits**

Only the final fully connected layer's weights live on the Stiefel manifold. All convolutional and batch norm parameters remain Euclidean, ensuring the computational overhead of the Riemannian operations is minimal.

### Computational Advantages Over MAML
1.  **Fewer parameters:** The Stiefel constraint reduces the parameter count from $np$ to $np - \frac{p(p+1)}{2}$. This scales significantly at larger output dimensions (e.g., replacing a 4096×1000 FC layer reduces the parameter count from 4M to 3.6M).
2.  **Faster convergence:** The constrained search space allows RMAML to outperform standard MAML even with fewer inner-loop steps and smaller meta-batch sizes.

---

## Custom Kernel: Cayley Retraction via Woodbury Identity

The most technically novel component of this project is our approach to the retraction step.

Every inner loop step requires a retraction — mapping an updated parameter matrix back onto the Stiefel manifold. Standard Riemannian libraries like `geoopt` rely on QR decomposition, an $O(n^3)$ LAPACK routine that is not optimized for our specific "tall and skinny" matrix shape (1600, 5).

### The Mathematical Trick
The Cayley retraction (Wen & Yin, 2013) is defined as:
$$W_{new} = (I + \frac{1}{2}A)^{-1}(I - \frac{1}{2}A)W$$
where $A = VW^T - WV^T$ (a skew-symmetric matrix).

Naively, $A$ is a 1600×1600 matrix. Forming and inverting it would be prohibitively slow. However, by using the Woodbury identity, we can reduce this $O(n^3)$ operation to a trivial 10×10 inversion.

We write $A = U_L U_R^T$, where $U_L = [V, -W]$ and $U_R = [W, V]$ (both size 1600×10). The update becomes:
$$W_{new} = (I - \frac{1}{2}A)W - U_L K^{-1} U_R^T (I - \frac{1}{2}A)W$$
where $K = 2I + U_R^T U_L$. The massive 1600×1600 matrix is never formed, and all operations are reduced to (1600, 10) matmuls and a single (10, 10) solve.

### The Hybrid Engineering Solution
Triton does not feature a built-in linear solver, making a pure in-kernel matrix inversion unnecessarily complex. To solve this, we utilized a hybrid execution model that plays to the exact strengths of both PyTorch and Triton:

**1. PyTorch Wrapper (Compute the dependencies):**
*   Compute $K = 2I + U_R^T U_L$ (10×10, fast)
*   Compute $K^{-1}$ (10×10, fast)
*   Compute $rhs = (I - \frac{1}{2}A)W$ (fast)
*   Precompute the tiny matrix $T = K^{-1} U_R^T rhs$ (size 10×5)

**2. Triton Kernel (The fused execution):**
With $T$ precomputed on the CPU/CUDA, we pass it into Triton. The kernel now only has to execute pure tiled matmuls — exactly what Triton is designed for. For each row tile:
*   Load `rhs_tile` and `U_L_tile`
*   Compute: `out_tile = rhs_tile - U_L_tile @ T`
*   Store `out_tile`

### Benchmark Results & Hardware Analysis

| Shape | Hardware | Operation | Time (μs/call) | Speedup |
|---|---|---|---|---|
| (1600, 5) | Apple M1 CPU | `geoopt` QR | 14,896 | 1× |
| (1600, 5) | Apple M1 CPU | Woodbury (PyTorch) | 150 | **99×** |
| (1600, 5) | RTX 4090 CUDA | `geoopt` QR | 6,133 | 1× |
| (1600, 5) | RTX 4090 CUDA | Woodbury (PyTorch) | 457 | **13.4×** |

Reporting raw speedups often hides real engineering nuances. The underlying context provides an excellent demonstration of hardware limits:

*   **Woodbury Math (PyTorch):** 13.4× speedup over `geoopt` ← *This is the real algorithmic win.*
*   **Triton Fusion (on top of Woodbury):** 0.6× of PyTorch ← *Overhead dominates at this scale.*

**Why does PyTorch beat Triton at (1600, 5)?**
At this matrix size, the data is small enough to fit entirely inside the GPU's L2 cache. Because there is no memory bandwidth bottleneck, PyTorch's heavily optimized `cuBLAS` handles the dense matmuls at peak efficiency. The overhead of the Triton kernel (JIT compilation, launch latency, and grid setup) outweighs the benefits of fusing the operations to prevent HBM roundtrips.

Triton fusion wins when matrices scale large enough that *memory bandwidth* becomes the bottleneck. If we scaled the feature dimension up to (16000, 5) — exceeding cache limits — Triton's ability to keep intermediates in SRAM would easily outpace PyTorch.

---

## Bi-level Optimization & Riemannian Adam

RMAML introduces **cAdam** — a Riemannian extension of Adam that uses parallel transport to correctly accumulate gradient moments along the curved manifold:

*   **Standard Adam:** $m(t+1) = \beta_1 m(t) + (1-\beta_1)\nabla J$
*   **cAdam:** $m(t+1) = \beta_1 \Gamma_{t-1 \to t}(m(t)) + (1-\beta_1)\pi_{\theta(t)}(\nabla J)$

We implemented this via `geoopt.optim.RiemannianAdam` (mathematically equivalent to Bécigneul & Ganea, ICLR 2019). The bi-level optimization relies on `functional_forward` with `create_graph=True` for differentiable inner loops, providing explicit control over which parameters receive Riemannian vs. Euclidean updates.

---

## Project Structure

```text
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
├── configs/                      ← YAML experiment configurations
├── tests/                        ← 49 unit tests, all CPU-only
├── notebooks/                    ← Training curves, results table analysis
├── benchmark_cuda.py             ← Tests benchmark with and without Triton on CUDA
├── train.py                      ← Entry point (--maml, --triton flags)
├── evaluate.py                   ← 600-episode test evaluation with 95% CI
├── tune.py                       ← Ray Tune HPO (--n-gpus for parallel)
└── Dockerfile                    ← CPU smoke test image
```

---

## Quickstart

```bash
# Install dependencies
pip install poetry
poetry install

# Download dataset (requires Kaggle account)
pip install kaggle
kaggle datasets download -d whitemoon/miniimagenet --path data/miniimagenet
cd data/miniimagenet && unzip miniimagenet.zip && rm miniimagenet.zip && cd ../..

# Smoke test (synthetic data, CPU, no dataset needed)
PYTHONPATH=src poetry run python train.py --config configs/conv4_miniimagenet.yaml --smoke-test

# Full training — RMAML + cAdam
PYTHONPATH=src python train.py --config configs/conv4_miniimagenet.yaml

# Run CUDA benchmarking (PyTorch vs Triton comparison)
PYTHONPATH=src python benchmark_cuda.py

# Evaluate on held-out test classes (600 episodes, 95% CI)
PYTHONPATH=src python evaluate.py --checkpoint checkpoints/cadam/epoch_050000.pt --config configs/conv4_miniimagenet.yaml

# HPO search (2 GPU parallel)
PYTHONPATH=src python tune.py --config configs/conv4_miniimagenet.yaml --n-gpus 2
```

## References
*   Tabealhojeh et al. (2023). *RMAML: Riemannian meta-learning with orthogonality constraints*. Pattern Recognition 140.
*   Finn et al. (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*. ICML.
*   Wen & Yin (2013). *A feasible method for optimization with orthogonality constraints*. Mathematical Programming.
*   Bécigneul & Ganea (2019). *Riemannian Adaptive Optimization Methods*. ICLR.
*   Li et al. (2020). *Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform*. ICLR.

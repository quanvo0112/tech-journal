# Training Diagnostics

- **Problem Link:** [NeetCode - Training Diagnostics](https://neetcode.io/problems/training-diagnostics)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `Training`
- **Implementation Framework:** `PyTorch`
- **Core Concept / Formulation:** `Neural Network Health Diagnostics, Activation Statistics (Mean, Std, Dead Fraction), Gradient Statistics (Mean, Std, L2 Norm), Multi-Stage Priority Decision Rules (Dead Neurons, Exploding Gradients, Vanishing Gradients, Healthy)`
- **Last Practiced:** 2026-10-09
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Diagnostic Metrics

### Overview & Motivation
Deep neural networks are susceptible to silent training pathologies—such as dying ReLU units, numerical explosion, and vanishing gradients—that cause optimization stagnation without raising runtime errors.

**Training Diagnostics** establishes an automated health monitoring suite that samples activation distributions on the forward pass and gradient magnitudes on the backward pass, applying a strict priority-ordered diagnostic rulebook to classify network health into:
$$\text{Health Status} \in \{\texttt{"dead\_neurons"}, \texttt{"exploding\_gradients"}, \texttt{"vanishing\_gradients"}, \texttt{"healthy"}\}$$

---

### Step-by-Step Diagnostic Metrics

#### 1. Activation Statistics (`compute_activation_stats`)
For each `nn.Linear` layer processing an activation tensor $X \in \mathbb{R}^{B \times D}$:
- **Activation Mean ($\mu$):**
  $$\mu = \frac{1}{B \cdot D} \sum_{i=1}^B \sum_{j=1}^D X_{i, j}$$
- **Activation Standard Deviation ($\sigma$):**
  $$\sigma = \sqrt{\frac{1}{B \cdot D} \sum_{i=1}^B \sum_{j=1}^D (X_{i, j} - \mu)^2}$$
- **Dead Neuron Fraction ($\rho_{\text{dead}}$):**
  A feature channel $j \in [1, D]$ is defined as **dead** if its activation is non-positive across every sample in the batch:
  $$\text{is\_dead}(j) = \bigwedge_{i=1}^B (X_{i, j} \le 0)$$
  The dead fraction represents the proportion of inactive channels across the layer:
  $$\rho_{\text{dead}} = \frac{1}{D} \sum_{j=1}^D \mathbb{I}\left[\forall i \in [1, B], X_{i, j} \le 0\right]$$
  Evaluated cleanly in PyTorch via `(x <= 0).all(dim=0).float().mean().item()`.

#### 2. Gradient Statistics (`compute_gradient_stats`)
Following a standard forward pass and MSE loss backpropagation:
- **Gradient Mean ($\mu_{\text{grad}}$):** Mean element of $\nabla_W \mathcal{L}$.
- **Gradient Standard Deviation ($\sigma_{\text{grad}}$):** Standard deviation of elements in $\nabla_W \mathcal{L}$.
- **Gradient Frobenius $L_2$ Norm ($\|\nabla_W \mathcal{L}\|_2$):**
  $$\|\nabla_W \mathcal{L}\|_2 = \sqrt{\sum_{i=1}^{D_{\text{out}}} \sum_{j=1}^{D_{\text{in}}} \left(\frac{\partial \mathcal{L}}{\partial W_{i, j}}\right)^2} = \text{torch.norm}(\nabla_W \mathcal{L})$$

---

### 3. Diagnostic Hierarchy & Priority Rulebook

The `diagnose()` routine applies a deterministic sequence of evaluation rules. The **first** triggered condition immediately dictates the final diagnostic verdict:

| Priority | Evaluation Condition | Classification Verdict | Diagnostic Rationale |
|:---:|:---|:---:|:---|
| **1** | Any layer has $\rho_{\text{dead}} > 0.5$ | `dead_neurons` | Over 50% of neurons in a layer are permanently inactive |
| **2** | Any layer has $\|\nabla_W \mathcal{L}\|_2 > 1000$ | `exploding_gradients` | Unbounded gradient expansion triggers numeric instability |
| **3** | Final layer has $\|\nabla_{W_{\text{last}}} \mathcal{L}\|_2 < 10^{-5}$ | `vanishing_gradients` | Output-adjacent gradient signal has collapsed near zero |
| **4a** | Any layer has activation $\sigma < 0.1$ | `vanishing_gradients` | Activation variance collapsed into degenerate constant |
| **4b** | Any layer has activation $\sigma > 10.0$ | `exploding_gradients` | Uncontrolled signal magnitude expansion across depth |
| **5** | No conditions violated | `healthy` | Normal activation scale and stable backpropagated gradients |

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input Batch | $X$ | `(B, D_in)` | Mini-batch of feature inputs |
| Target Batch | $Y$ | `(B, D_out)` | Ground-truth regression targets |
| Layer Activations | $X^{(l)}$ | `(B, D_l)` | Intermediate activations produced by $l$-th layer |
| Dead Channel Mask | $M_{\text{dead}}$ | `(D_l,)` | Boolean vector: `True` if channel $\le 0$ for all $B$ samples |
| Layer Weight Matrix | $W^{(l)}$ | `(D_out, D_in)` | Trainable parameter tensor |
| Weight Gradient | $\nabla_{W^{(l)}} \mathcal{L}$ | `(D_out, D_in)` | Analytical loss derivative matrix |

---

## 3. Vectorized Implementation & Numerical Stability

1. **Context-Managed Evaluation (`torch.no_grad()`):**
   - Activation inspection does not compute gradients. Wrapping forward traversal inside `with torch.no_grad():` suppresses autograd computational graph allocations, saving memory and compute.
2. **Gradient Isolation (`model.zero_grad()`):**
   - PyTorch accumulates gradients in `.grad` attributes by default. Calling `model.zero_grad()` before the backward pass guarantees that monitored gradient norms reflect strictly the current sample batch.
3. **Batch-Invariant Channel Reduction (`all(dim=0)`):**
   - Evaluating dead neurons requires testing that all batch instances are simultaneously non-positive. Reducing across dimension 0 produces a 1D boolean mask of shape `(D,)`, whose mean gives the exact fraction of dead channels without nested loops.

---

## 4. Complexity Analysis

Let $B$ denote the mini-batch size, $L$ the number of linear layers, and $N_P$ the total parameters:

- **Time Complexity:** $\mathcal{O}(B \cdot N_P)$
  - Activation pass evaluates each layer sequentially: $\mathcal{O}(B \cdot N_P)$.
  - Gradient computation performs one standard backward pass: $\mathcal{O}(B \cdot N_P)$.
  - Statistical calculations (`mean`, `std`, `norm`) are linear in layer dimensions: $\mathcal{O}(N_P)$.
  - Overall time complexity is equivalent to a single standard training iteration.
- **Space Complexity:** $\mathcal{O}(B \cdot D_{\max} + N_P)$
  - Activations require temporary buffers proportional to the largest hidden layer $B \cdot D_{\max}$.
  - Weight gradients occupy memory identical to model parameters $N_P$.

---

## 5. Edge Cases & Gotchas

- [x] **1D vs. 2D Activation Tensors:** If an activation tensor is 1D (e.g. `(D,)` without batch dimension), evaluating `.all(dim=0)` collapses the channel axis. Guarding with `if x.dim() >= 2` ensures correct handling across single-sample and batched inputs.
- [x] **Strict Priority Ordering:** If a model exhibits both dead neurons ($\rho_{\text{dead}} > 0.5$) and exploding gradients ($\|\nabla W\| > 1000$), the function must return `"dead_neurons"` because Priority 1 precedes Priority 2.
- [x] **Precision Rounding:** All reported numerical metrics in dictionaries must be rounded to 4 decimal places via `round(..., 4)`.

---

## 6. Clean Code

```python
from typing import Dict, List
import torch
import torch.nn as nn


class Solution:

    def compute_activation_stats(
        self, model: nn.Module, x: torch.Tensor
    ) -> List[Dict[str, float]]:
        stats = []

        # 1. Forward pass without tracking autograd graph
        with torch.no_grad():
            for module in model.children():
                x = module(x)

                if isinstance(module, nn.Linear):
                    mean_val = round(x.mean().item(), 4)
                    std_val = round(x.std().item(), 4)

                    # Compute dead neuron fraction (channels non-positive across all samples)
                    if x.dim() >= 2:
                        dead_frac = (x <= 0).all(dim=0).float().mean().item()
                    else:
                        dead_frac = (x <= 0).float().mean().item()

                    stats.append({
                        "mean": mean_val,
                        "std": std_val,
                        "dead_fraction": round(dead_frac, 4),
                    })

        return stats

    def compute_gradient_stats(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> List[Dict[str, float]]:
        # 2. Reset gradients and perform backpropagation
        model.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        stats = []

        # 3. Extract gradient statistics for each linear layer
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4),
                })

        return stats

    def diagnose(
        self,
        activation_stats: List[Dict[str, float]],
        gradient_stats: List[Dict[str, float]],
    ) -> str:
        # Priority 1: Dead neurons
        for stats in activation_stats:
            if stats["dead_fraction"] > 0.5:
                return "dead_neurons"

        # Priority 2: Exploding gradients
        for stats in gradient_stats:
            if stats["norm"] > 1000:
                return "exploding_gradients"

        # Priority 3: Vanishing gradients (final layer gradient norm check)
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # Priority 4: Abnormal activation standard deviation
        for stats in activation_stats:
            if stats["std"] < 0.1:
                return "vanishing_gradients"
            if stats["std"] > 10.0:
                return "exploding_gradients"

        # Priority 5: Healthy
        return "healthy"
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Diagnostic Health Pipeline

```text
               Input Batch (X, Y)
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
Forward Pass (no_grad)          Forward + Backward (MSE)
       │                                 │
Extract Activation Stats:       Extract Gradient Stats:
 - mean, std                     - mean, std
 - dead_fraction                 - L2 norm
       └────────────────┬────────────────┘
                        ▼
             Priority-Based Diagnosis
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
    Rule Violations?           No Violations
          │                           │
          ├─ dead_fraction > 0.5 ────> "dead_neurons"
          ├─ grad_norm > 1000 ───────> "exploding_gradients"
          ├─ last_norm < 1e-5 ───────> "vanishing_gradients"
          ├─ act_std < 0.1 ──────────> "vanishing_gradients"
          └─ act_std > 10.0 ─────────> "exploding_gradients"
                                      │
                                      ▼
                                  "healthy"
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Tracking activation distributions and gradient Frobenius norms allows deterministic early detection of dying ReLUs, gradient explosion, and signal vanishing before training destabilizes.


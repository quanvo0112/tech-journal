# Weight Initialization

- **Problem Link:** [NeetCode - Weight Initialization](https://neetcode.io/problems/weight-initialization)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `Build a Neural Net`
- **Implementation Framework:** `PyTorch`
- **Core Concept / Formulation:** `Weight Initialization Schemes, Xavier/Glorot Normal, Kaiming/He Normal, Activation Variance Propagation, Exploding & Vanishing Activations Diagnostic`
- **Last Practiced:** 2026-10-06
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Motivation
Weight initialization is the process of defining initial numerical values for trainable weight parameters prior to gradient-based optimization. If weights are initialized poorly:
- **Exploding Activations:** When initial weights have excessive variance, signals compound exponentially across consecutive layers, resulting in numerical overflow (`NaN` / `Inf`) and destabilizing backpropagation gradients.
- **Vanishing Activations:** When initial weights are scaled too small, activation variance shrinks toward zero with depth, causing neuron saturation or zero signals, effectively deadening learning in early layers.

The principal objective of modern initialization schemes is to **preserve activation and gradient variance** across deep layer stacks.

---

### 1. Xavier / Glorot Initialization
Introduced by Xavier Glorot and Yoshua Bengio (2010), this technique is tailored for symmetric activations around zero with unit derivative at the origin (e.g., Sigmoid, $\tanh$, Linear).

Under **Xavier Normal Initialization**, weights are drawn from a zero-mean Gaussian distribution whose standard deviation balances input and output connectivity:
$$W \sim \mathcal{N}\left(0, \sigma^2\right), \quad \sigma = \sqrt{\frac{2}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}$$

where:
- $\text{fan}_{\text{in}}$: number of input units to the layer ($D_{\text{in}}$).
- $\text{fan}_{\text{out}}$: number of output units from the layer ($D_{\text{out}}$).

By scaling with $\sqrt{2 / (\text{fan}_{\text{in}} + \text{fan}_{\text{out}})}$, the variance of the forward activations and the backward gradients remain approximately constant from layer to layer.

---

### 2. Kaiming / He Initialization
Introduced by Kaiming He et al. (2015), this method is specifically designed for non-linear rectifier units ($\text{ReLU}$).

Because $\text{ReLU}(z) = \max(0, z)$ zeroes out roughly half of the activation distribution (all negative pre-activations), it effectively halves the variance of the transmitted signal. Xavier initialization fails here because the signal steadily shrinks across layers. Kaiming initialization compensates by incorporating a factor of 2 in the numerator scaled solely by input fan-in:
$$W \sim \mathcal{N}\left(0, \sigma^2\right), \quad \sigma = \sqrt{\frac{2}{\text{fan}_{\text{in}}}}$$

This extra scaling factor maintains unit variance in the activations even after passing through the zeroing threshold of $\text{ReLU}$.

---

### 3. Standard Random Gaussian Initialization (Baseline)
When weights are drawn from an unscaled standard normal distribution:
$$W \sim \mathcal{N}(0, 1), \quad \sigma = 1.0$$
Each linear layer computes a sum of $\text{fan}_{\text{in}}$ independent random variables. The variance of the pre-activation scales linearly with $\text{fan}_{\text{in}}$ ($\text{Var}(z) \approx \text{fan}_{\text{in}} \cdot \text{Var}(x)$). Over $L$ layers, activations compound geometrically, causing catastrophic **activation explosion**.

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Layer Dimensions | `dims` | `[input_dim] + [hidden_dim] * L` | Sequence of dimensionalities from input through hidden layers |
| Weight Matrix $l$ | $W_l$ | `(dims[l+1], dims[l])` | PyTorch linear layer convention: `(fan_out, fan_in)` |
| Input Batch | $x$ | `(1, input_dim)` | Batch size of 1 with initial standard normal features |
| Linear Projection | $x W_l^T$ | `(1, dims[l+1])` | Batched matrix multiplication: `x @ w.T` |
| Activated State | $\text{ReLU}(x W_l^T)$ | `(1, dims[l+1])` | Post-activation state forwarded to subsequent layer |

---

## 3. Implementation Invariants & Numerical Stability

1. **Weight Orientation & Batched Matrix Multiplication (`x @ w.T`):**
   - In PyTorch linear layers, weights are shaped as `(out_features, in_features)` $\equiv (\text{fan}_{\text{out}}, \text{fan}_{\text{in}})$.
   - For an input row vector $x \in \mathbb{R}^{B \times \text{fan}_{\text{in}}}$, the affine projection requires transposing the weight matrix:
     $$\text{output} = x \mathbin{@} W^T \in \mathbb{R}^{B \times \text{fan}_{\text{out}}}$$
   - Contrast with the column-first convention ($W x$) or row-first without transpose ($x \mathbin{@} W$ when $W$ is $(\text{in}, \text{out})$).

2. **Pseudo-Random Number Generator Seed Ordering Invariant:**
   - PyTorch shares a single global pseudo-random state per execution context when seeded via `torch.manual_seed(0)`.
   - In `check_activations()`, **all weight matrices across all layers must be sampled first** in loop order $0, 1, \dots, L-1$.
   - The input tensor $x \sim \mathcal{N}(0, 1)$ must be generated **after** all layer weights are fully materialized. Altering this sequence draws completely different random vectors and fails deterministic test assertions.

3. **Standard Deviation Diagnostic Metric:**
   - Calculating `x.std().item()` after each $\text{ReLU}$ stage directly quantifies signal dispersion.
   - Stable standard deviation ($\approx 0.5 - 1.5$) confirms healthy propagation.
   - Explosive standard deviation (e.g., growing from $4.06 \to 2878.09$) highlights unconstrained variance amplification.

---

## 4. Complexity Analysis

- **Time Complexity:**
  - `xavier_init` / `kaiming_init`: $\mathcal{O}(\text{fan}_{\text{in}} \cdot \text{fan}_{\text{out}})$ to sample and scale Gaussian matrix elements.
  - `check_activations`: $\mathcal{O}(L \cdot \text{hidden\_dim}^2)$ for $L$ linear projections of dimension $\text{hidden\_dim} \times \text{hidden\_dim}$ with batch size 1.
- **Space / Activation Memory:**
  - $\mathcal{O}(L \cdot \text{hidden\_dim}^2)$ auxiliary memory to store all allocated weight tensors before the forward pass.

---

## 5. Edge Cases & Gotchas

- [x] **Seed Synchronization:** Always invoke `torch.manual_seed(0)` at the beginning of each initialization method to ensure identical, reproducible outputs across test executions.
- [x] **Rounding Protocol:** Individual initialization matrices are rounded to 4 decimals (`torch.round(weights, decimals=4).tolist()`), whereas activation standard deviations in `check_activations()` are rounded to 2 decimals (`round(x.std().item(), 2)`).
- [x] **Fan Dimension Mapping:** In `check_activations()`, layer $i$ maps $\text{dims}[i] \to \text{dims}[i+1]$. Thus, $\text{fan}_{\text{in}} = \text{dims}[i]$ and $\text{fan}_{\text{out}} = \text{dims}[i+1]$. For Kaiming initialization, $\sigma = \sqrt{2.0 / \text{dims}[i]}$; for Xavier, $\sigma = \sqrt{2.0 / (\text{dims}[i] + \text{dims}[i+1])}$.

---

## 6. Clean Code

```python
import torch
import torch.nn as nn
import math


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> list[list[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> list[list[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str
    ) -> list[float]:

        torch.manual_seed(0)

        # Construct layer dimensional sequence
        dims = [input_dim] + [hidden_dim] * num_layers

        weights = []

        # Sample all layer weights sequentially before generating input x
        for i in range(num_layers):
            if init_type == "xavier":
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == "kaiming":
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0

            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)

        # Generate test input batch (1, input_dim)
        x = torch.randn(1, input_dim)

        stds = []

        # Forward pass through linear layers followed by ReLU
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Initialization Schemes Comparison

| Initialization | Standard Deviation ($\sigma$) | Theoretical Target | Primary Use Case |
| :--- | :---: | :--- | :--- |
| **Random** | $1.0$ | No scaling | Baseline; causes rapid activation explosion |
| **Xavier / Glorot** | $\sqrt{\frac{2}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}$ | Preserves variance across linear/symmetric activations | $\tanh$, Sigmoid, Linear projections |
| **Kaiming / He** | $\sqrt{\frac{2}{\text{fan}_{\text{in}}}}$ | Compensates for half-spectrum zeroing in ReLU | $\text{ReLU}$, Leaky $\text{ReLU}$, GeLU |

---

### Activation Dispersion Diagnostic Flow

```text
               check_activations(num_layers=5, dim=64)
                                  ↓
                        torch.manual_seed(0)
                                  ↓
                   Generate All Weights W1 ... W5
                                  ↓
                       Generate Input x (1, 64)
                                  ↓
              ┌───────────────────┴───────────────────┐
              │ For each layer w in weights:          │
              │   1. x = x @ w.T                      │
              │   2. x = relu(x)                      │
              │   3. stds.append(round(x.std(), 2))   │
              └───────────────────┬───────────────────┘
                                  ↓
                       Activation Std Trajectory:
   Random:  [4.06,  26.17, 126.32, 695.77, 2878.09]  <-- Exploding!
   Kaiming: [0.78,   0.81,   0.85,   0.82,    0.79]  <-- Stable variance!
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Correct weight initialization scales variance by layer fan-in to prevent exponential signal explosion or collapse, forming the mathematical foundation for stable deep network training.

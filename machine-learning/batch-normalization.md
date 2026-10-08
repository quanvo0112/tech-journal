# Batch Normalization

- **Problem Link:** [NeetCode - Batch Normalization](https://neetcode.io/problems/batch-normalization)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `PyTorch`
- **Implementation Framework:** `NumPy`
- **Core Concept / Formulation:** `Batch Normalization (BatchNorm), Feature-Wise Batch Reduction (axis=0), Exponential Moving Average (Running Statistics), Training vs. Inference Discrepancy, Affine Scale and Shift`
- **Last Practiced:** 2026-10-07
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Motivation
Introduced by Sergey Ioffe and Christian Szegedy (2015), **Batch Normalization** addresses internal covariate shift by standardizing each feature channel across the mini-batch dimension.

Unlike Layer Normalization (which normalizes across feature dimensions per sample), Batch Normalization normalizes **across samples for each individual feature dimension**. This stabilizes gradient flows, permits substantially higher learning rates, and acts as a mild regularizer during training.

---

### Dual-Phase Operational Dynamics

Given an input batch matrix $X \in \mathbb{R}^{N \times D}$ ($N$ samples, $D$ features):

#### Phase 1: Training Mode (`training = True`)
1. **Batch Mean across Samples (`axis=0`):**
   $$\mu_B = \frac{1}{N} \sum_{i=1}^N X_{i, \ast} \in \mathbb{R}^D$$

2. **Batch Population Variance (`axis=0`):**
   $$\sigma_B^2 = \frac{1}{N} \sum_{i=1}^N (X_{i, \ast} - \mu_B)^2 \in \mathbb{R}^D$$

3. **Standardization with Epsilon Safeguard:**
   $$\hat{X} = \frac{X - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

4. **Running Statistics Tracking (Exponential Moving Average):**
   To prepare for inference, running statistics accumulate batch historical moments:
   $$\text{running\_mean}_{\text{new}} = (1 - \text{momentum}) \cdot \text{running\_mean} + \text{momentum} \cdot \mu_B$$
   $$\text{running\_var}_{\text{new}} = (1 - \text{momentum}) \cdot \text{running\_var} + \text{momentum} \cdot \sigma_B^2$$

#### Phase 2: Inference Mode (`training = False`)
During evaluation or test inference, predictions for a single sample must not depend on other samples in the batch. Thus, batch statistics are bypassed entirely in favor of frozen running statistics:
$$\hat{X} = \frac{X - \text{running\_mean}}{\sqrt{\text{running\_var} + \epsilon}}$$

#### Learnable Affine Transform (Both Phases)
In both training and inference modes, normalized activations pass through an affine projection to preserve model representational capacity:
$$Y = \gamma \odot \hat{X} + \beta$$
where $\gamma \in \mathbb{R}^D$ is the scale parameter and $\beta \in \mathbb{R}^D$ is the shift parameter.

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input Batch | $X$ | `(N, D)` | 2D matrix of $N$ samples across $D$ features |
| Batch Mean | $\mu_B$ | `(D,)` | Mean of each feature column across all $N$ rows |
| Batch Variance | $\sigma_B^2$ | `(D,)` | Variance of each feature column across all $N$ rows |
| Running Mean | $\mu_{\text{run}}$ | `(D,)` | Accumulated historical running mean vector |
| Running Variance | $\sigma^2_{\text{run}}$ | `(D,)` | Accumulated historical running variance vector |
| Scale Parameter | $\gamma$ | `(D,)` | Learnable feature scaling vector |
| Shift Parameter | $\beta$ | `(D,)` | Learnable feature shifting vector |
| Output Batch | $Y$ | `(N, D)` | Final affine-projected normalized batch |

---

## 3. Vectorized Implementation & Key Invariants

1. **Why `axis=0` (Reduction Axis):**
   - In a 2D matrix `(N, D)`, row index $0$ indexes samples and column index $1$ indexes features.
   - To normalize each feature channel independently, we reduce along the sample dimension:
     $$\text{batch\_mean} = \text{np.mean}(x, \text{axis}=0) \in \mathbb{R}^D$$
   - NumPy broadcasting automatically stretches the resulting `(D,)` vector across all $N$ rows during `(x - batch_mean)`.

2. **BatchNorm vs. LayerNorm Architectural Invariant:**
   - **BatchNorm (`axis=0`):** Normalizes across samples $N$ for each feature $D$. Dependent on mini-batch statistics.
   - **LayerNorm (`axis=1` or across features):** Normalizes across feature channels $D$ for each independent sample $N$. Completely batch-size independent.

3. **PyTorch Momentum Convention Gotcha:**
   - In deep learning optimizers (e.g., SGD Momentum), momentum $\beta \approx 0.9$ represents weight on the *past* velocity.
   - However, in PyTorch and NeetCode BatchNorm, the parameter `momentum` represents the weight assigned to the **new batch observation** (often set to $0.1$):
     $$\text{running}_{\text{new}} = (1 - \text{momentum}) \cdot \text{running}_{\text{old}} + \text{momentum} \cdot \text{batch}$$
   - Inverting these coefficients causes catastrophic test failures.

---

## 4. Complexity Analysis

Let $N$ denote batch size and $D$ denote the number of features:

- **Time Complexity:** $\mathcal{O}(N \cdot D)$
  - Computing column-wise mean and variance requires a single pass over all $N \cdot D$ elements: $\mathcal{O}(N \cdot D)$.
  - Element-wise normalization and affine projection operate over the $N \times D$ grid: $\mathcal{O}(N \cdot D)$.
  - Updating running statistics takes $\mathcal{O}(D)$ operations.
  - Overall time complexity is strictly $\mathcal{O}(N \cdot D)$.
- **Space Complexity:** $\mathcal{O}(N \cdot D)$
  - Memory buffers for intermediate standardized matrix $\hat{X}$ and output matrix $Y$.

---

## 5. Edge Cases & Gotchas

- [x] **Inference Independence:** Never compute `np.mean(x, axis=0)` or `np.var(x, axis=0)` during inference (`training=False`). Doing so leaks mini-batch statistics into evaluation and breaks single-sample inference ($N=1$, where sample variance would be 0).
- [x] **Epsilon Safeguard:** $\epsilon > 0$ under the square root prevents division by zero when batch variance or running variance evaluates to zero.
- [x] **Rounding Protocol:** Output tensors and updated running statistics are rounded to 4 decimal places via `np.round(..., 4).tolist()`.

---

## 6. Clean Code

```python
import numpy as np
from typing import Tuple, List


class Solution:

    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool
    ) -> Tuple[List[List[float]], List[float], List[float]]:

        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:
            # 1. Compute batch statistics across samples (axis=0)
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # 2. Normalize using current batch moments
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)

            # 3. Update exponential moving average running statistics
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var = (1 - momentum) * running_var + momentum * batch_var
        else:
            # Inference mode: standardize using precomputed population statistics
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        # 4. Apply learnable affine scale and shift
        y = gamma * x_hat + beta

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Dual-Mode Execution Graph

```text
                        Input Batch X (N, D)
                                  │
                       training == True?
                        /               \
                   Yes /                 \ No
                      ▼                   ▼
          batch_mean = mean(x, axis=0)    x_hat = (x - running_mean)
          batch_var  = var(x, axis=0)             / sqrt(running_var + eps)
          x_hat      = (x - batch_mean)
                       / sqrt(batch_var + eps)
                      │
          Update Running Stats:
          run_mean = (1 - m)*run_mean + m*batch_mean
          run_var  = (1 - m)*run_var  + m*batch_var
                      │
                      └─────────┬─────────┘
                                ▼
                        y = gamma * x_hat + beta
                                ▼
                        Output Batch Y (N, D)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** BatchNorm standardizes across samples along each feature column (`axis=0`) during training while tracking running statistics via $(1 - \text{momentum}) \cdot \text{running} + \text{momentum} \cdot \text{batch}$ to guarantee deterministic inference.


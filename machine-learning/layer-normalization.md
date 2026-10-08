# Layer Normalization

- **Problem Link:** [NeetCode - Layer Normalization](https://neetcode.io/problems/layer-normalization)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `PyTorch`
- **Implementation Framework:** `NumPy`
- **Core Concept / Formulation:** `Layer Normalization (LayerNorm), Feature-Wise Normalization, Affine Transformation (Scale & Shift), Numerical Stability Epsilon, Population Variance`
- **Last Practiced:** 2026-10-07
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Motivation
Introduced by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton (2016), **Layer Normalization** standardizes the activations of a neural network across the feature dimension for an individual sample, ensuring stable forward signal propagation and balanced gradient magnitudes during backpropagation.

Unlike **Batch Normalization** (which normalizes across batch instances and struggles with small batch sizes, sequence modeling, and distributed setups), Layer Normalization computes its statistics **independently for each individual sample**. This property makes it the foundational normalization layer across Transformer architectures (Pre-LN and Post-LN Transformers, BERT, GPT).

---

### Step-by-Step Formulation

Given an input feature vector $x \in \mathbb{R}^d$:

1. **Feature Mean:**
   $$\mu = \frac{1}{d} \sum_{i=1}^d x_i = \text{mean}(x)$$

2. **Population Variance:**
   $$\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2 = \text{var}(x)$$

3. **Standardization with Epsilon Safeguard:**
   $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
   Where $\epsilon = 10^{-5}$ is a small positive constant added to prevent division by zero when feature variance is near-zero or all features are identical.

4. **Learnable Affine Transformation (Scale & Shift):**
   $$y = \gamma \odot \hat{x} + \beta$$
   - $\gamma \in \mathbb{R}^d$ (**scale / gain**): Initialized to $1$, allows the model to adaptively restore or rescale feature variances if identity scaling is not optimal.
   - $\beta \in \mathbb{R}^d$ (**shift / bias**): Initialized to $0$, allows the model to learn an adaptive baseline offset for each feature dimension.

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input Features | $x$ | `(d,)` | 1D feature activation vector |
| Scalar Mean | $\mu$ | `()` | Mean scalar computed across all features |
| Scalar Variance | $\sigma^2$ | `()` | Population variance scalar computed across features |
| Normalized Activations | $\hat{x}$ | `(d,)` | Standardized activations with zero mean and unit variance |
| Scale Parameter | $\gamma$ | `(d,)` | Learnable feature scaling vector |
| Shift Parameter | $\beta$ | `(d,)` | Learnable feature shifting vector |
| Output Activations | $y$ | `(d,)` | Final affine-projected normalized vector |

---

## 3. Vectorized Implementation & Numerical Stability

1. **LayerNorm vs. BatchNorm vs. RMSNorm:**
   - **Batch Normalization:** Normalizes across the batch axis $B$ for each feature channel $C$. Dependent on batch size; unusable during autoregressive single-token generation.
   - **Layer Normalization:** Normalizes across feature dimensions $D$ for each individual token / sample. Completely independent of batch size.
   - **RMS Normalization (RMSNorm):** Simplifies LayerNorm by omitting mean subtraction ($\mu = 0$) and scaling solely by root-mean-square feature energy.

2. **Numerical Stability Safeguard ($\epsilon$):**
   - If an input vector contains identical elements (e.g., $x = [2.0, 2.0, 2.0]$), its variance is strictly $\sigma^2 = 0$.
   - Without $\epsilon$, evaluating $\frac{x - \mu}{\sqrt{0}}$ produces `NaN` / division by zero.
   - Adding $\epsilon = 10^{-5}$ under the radical ($\sqrt{\sigma^2 + \epsilon}$) guarantees finite, stable values.

---

## 4. Complexity Analysis

Let $d = \text{len}(x)$ denote the number of features in the vector:

- **Time Complexity:** $\mathcal{O}(d)$
  - Computing the mean requires one linear pass: $\mathcal{O}(d)$.
  - Computing the variance requires one linear pass: $\mathcal{O}(d)$.
  - Element-wise normalization and affine projection each take $\mathcal{O}(d)$ operations.
  - Overall time complexity is strictly $\mathcal{O}(d)$.
- **Space Complexity:** $\mathcal{O}(d)$
  - Vectorized intermediate allocations ($\hat{x}$, output vector) allocate buffers proportional to feature dimension $d$.

---

## 5. Edge Cases & Gotchas

- [x] **Population Variance (`ddof=0`):** NumPy's `np.var()` defaults to `ddof=0` ($\frac{1}{N} \sum (x_i - \mu)^2$), which exactly matches the population variance formula used by standard deep learning normalization layers. Do not use sample variance (`ddof=1`).
- [x] **Identical Values:** When all features have the same scalar value, $\sigma^2 = 0$ and $x - \mu = 0$. Standardization evaluates to $\frac{0}{\sqrt{0 + \epsilon}} = 0$, producing output $\beta$.
- [x] **Rounding Precision:** The prompt specifies rounding the output tensor to 5 decimal places via `np.round(out, 5)`.

---

## 6. Clean Code

```python
import numpy as np
from numpy.typing import NDArray


class Solution:

    def forward(
        self,
        x: NDArray[np.float64],
        gamma: NDArray[np.float64],
        beta: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        eps = 1e-5

        # 1. Compute feature-wise mean and variance
        mean = np.mean(x)
        var = np.var(x)

        # 2. Normalize to zero mean and unit variance with epsilon protection
        x_hat = (x - mean) / np.sqrt(var + eps)

        # 3. Apply learnable affine scale (gamma) and shift (beta)
        out = gamma * x_hat + beta

        return np.round(out, 5)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Computational Flow Graph

```text
               Input Feature Vector x (d,)
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
          mean = mean(x)            var = var(x)
               └────────────┬────────────┘
                            ▼
            x_hat = (x - mean) / sqrt(var + eps)
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
            gamma * x_hat          + beta
                  └─────────┬─────────┘
                            ▼
               Output Tensor y (d,)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Layer Normalization standardizes activations across the feature dimension independently per sample, while learnable affine parameters $\gamma$ and $\beta$ preserve the network's expressive capacity to scale and shift representations.


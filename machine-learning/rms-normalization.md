# RMS Normalization

- **Problem Link:** [NeetCode - RMS Normalization](https://neetcode.io/problems/rms-normalization)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `PyTorch`
- **Implementation Framework:** `NumPy`
- **Core Concept / Formulation:** `Root Mean Square Layer Normalization (RMSNorm), Magnitude Normalization, Scale Invariance, Mean-Free Normalization, Modern Transformer Architectures (LLaMA, Mistral, Gemma)`
- **Last Practiced:** 2026-10-08
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Motivation
Proposed by Biao Zhang and Rico Sennrich (2019) in *"Root Mean Square Layer Normalization"*, **RMSNorm** is an efficient reformulation of Layer Normalization designed to reduce computational overhead in deep neural networks.

The core insight behind RMSNorm is that the primary benefit of Layer Normalization comes from **rescaling the activations (scaling invariance)** rather than from mean-centering (shift invariance). By removing the mean subtraction step ($\mu = 0$) and omitting the additive bias vector $\beta$, RMSNorm reduces compute costs by approximately 10% to 50% while preserving identical training dynamics and representation stability.

Due to this computational efficiency, RMSNorm is now the default normalization choice in modern open-weight LLMs, including **LLaMA 1/2/3, Mistral, Mixtral, Gemma, Qwen, and DeepSeek**.

---

### Step-by-Step Formulation

Given an input feature vector $x \in \mathbb{R}^n$:

1. **Root Mean Square (RMS) with Epsilon Safeguard:**
   $$\text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2 + \epsilon} = \sqrt{\text{mean}(x^2) + \epsilon}$$
   Where $\epsilon > 0$ is a small scalar constant added inside the square root to guarantee numerical stability and prevent division by zero.

2. **Standardization by RMS:**
   $$\hat{x}_i = \frac{x_i}{\text{RMS}(x)} \iff \hat{x} = \frac{x}{\sqrt{\frac{1}{n} \|x\|_2^2 + \epsilon}}$$
   Notice that $\hat{x}$ has a root mean square of approximately $1$.

3. **Learnable Linear Scaling (Gain $\gamma$):**
   $$y_i = \gamma_i \hat{x}_i \iff y = \gamma \odot \hat{x}$$
   Where $\gamma \in \mathbb{R}^n$ is a learnable scaling parameter vector that allows the model to scale individual feature channels adaptively. Unlike LayerNorm, there is **no learnable additive shift parameter $\beta$**.

---

## 2. Structural Comparison: LayerNorm vs. RMSNorm

| Feature | Layer Normalization (LayerNorm) | Root Mean Square Normalization (RMSNorm) |
|:---|:---|:---|
| **Mean Centering** | Yes: $\mu = \frac{1}{n} \sum x_i$ subtracted from $x$ | **No**: Assumes mean is already sufficiently controlled |
| **Normalizing Factor** | Standard Deviation: $\sqrt{\sigma^2 + \epsilon} = \sqrt{\text{var}(x) + \epsilon}$ | Root Mean Square: $\sqrt{\text{mean}(x^2) + \epsilon}$ |
| **Shift Parameter ($\beta$)** | Yes: Additive $+ \beta$ vector | **No**: No additive bias offset |
| **Mathematical Equation** | $y = \gamma \odot \left(\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta$ | $y = \gamma \odot \left(\frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}\right)$ |
| **Computational Passes** | 2 memory read/write passes (mean then variance) | **1 single pass** over activations ($x^2$) |
| **Primary Adoption** | Classic Transformers, BERT, Original GPT-2 / GPT-3 | Modern LLMs: LLaMA 1/2/3, Mistral, Gemma, DeepSeek |

---

## 3. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input Features | $x$ | `(n,)` | 1D feature activation vector |
| Scalar RMS | $\text{RMS}(x)$ | `()` | Root-mean-square energy scalar computed across all features |
| Normalized Features | $\hat{x}$ | `(n,)` | Normalized feature vector scaled to unit RMS |
| Scale Parameter | $\gamma$ | `(n,)` | Learnable feature scaling vector |
| Output Vector | $y$ | `(n,)` | Final scaled normalized activation vector |

---

## 4. Vectorized Implementation & Numerical Stability

1. **Why Pure Vectorization (No Python Loops):**
   - Evaluating `np.mean(x ** 2)` executes via SIMD vector registers in C/Fortran speed, computing sum of squares in a single contiguous memory scan.
   - Vectorized division `x / rms` and element-wise scaling `gamma * x_hat` avoid slow interpreter loop overhead.

2. **Numerical Stability Safeguard ($\epsilon$):**
   - If an input vector contains all zeros ($x = [0.0, 0.0, \dots, 0.0]$), $\text{mean}(x^2) = 0$.
   - Without $\epsilon$, evaluating $\frac{x}{0}$ produces `NaN` / division by zero.
   - Adding $\epsilon$ under the radical ($\sqrt{\text{mean}(x^2) + \epsilon}$) ensures safe computation.

---

## 5. Complexity Analysis

Let $n = \text{len}(x)$ denote the number of features in the input vector:

- **Time Complexity:** $\mathcal{O}(n)$
  - Computing $x^2$ and its mean requires a single linear pass: $\mathcal{O}(n)$.
  - Square root and addition with $\epsilon$ take $\mathcal{O}(1)$.
  - Element-wise vector division $x / \text{rms}$ and scaling $\gamma \odot \hat{x}$ take $\mathcal{O}(n)$ operations.
  - Overall time complexity is strictly $\mathcal{O}(n)$, with significantly fewer arithmetic instructions and memory passes than LayerNorm.
- **Space Complexity:** $\mathcal{O}(n)$
  - Temporary array allocations for NumPy conversions (`x`, `gamma`, $\hat{x}$, `output`) allocate buffers proportional to feature dimension $n$.

---

## 6. Edge Cases & Gotchas

- [x] **Zero Vector Input:** When $x = [0.0, \dots, 0.0]$, $\text{mean}(x^2) = 0$, so $\text{rms} = \sqrt{\epsilon}$. Thus $\hat{x} = 0 / \sqrt{\epsilon} = 0$, producing output $0$.
- [x] **Negative Values:** Because all elements are squared ($x_i^2$), negative values contribute positively to the magnitude measure $\text{RMS}(x)$, preserving sign in $\hat{x} = x / \text{rms}$.
- [x] **Rounding Precision:** The prompt specifies rounding the output tensor to 4 decimal places via `np.round(output, 4).tolist()`.

---

## 7. Clean Code

```python
from typing import List
import numpy as np


class Solution:

    def rms_norm(
        self, x: List[float], gamma: List[float], eps: float
    ) -> List[float]:
        # 1. Convert inputs to NumPy arrays
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)

        # 2. Compute Root Mean Square (RMS) with epsilon stabilization
        rms = np.sqrt(np.mean(x**2) + eps)

        # 3. Scale input by 1 / RMS
        x_hat = x / rms

        # 4. Apply learnable channel gain (gamma) without additive bias
        output = gamma * x_hat

        # 5. Round to 4 decimal places and return as Python list
        return np.round(output, 4).tolist()
```

---

## 8. Architecture Walkthrough & Key Takeaways

### Computational Flow Graph

```text
               Input Feature Vector x (n,)
                             │
                             ├──────────────────────────┐
                             ▼                          │
                     x_squared = x ** 2                 │
                             │                          │
                             ▼                          │
                   mean_sq = mean(x_squared)            │
                             │                          │
                             ▼                          │
                  rms = sqrt(mean_sq + eps)             │
                             │                          │
                             └────────────┬─────────────┘
                                          ▼
                                     x_hat = x / rms
                                          │
                                          ├─────────────┐
                                          ▼             ▼
                                        gamma        x_hat
                                          └──────┬──────┘
                                                 ▼
                                        output = gamma * x_hat
                                                 │
                                                 ▼
                                     np.round(output, 4).tolist()
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** RMSNorm simplifies Layer Normalization by omitting mean-centering and additive bias ($\beta$), standardizing solely by root-mean-square feature energy ($\sqrt{\text{mean}(x^2) + \epsilon}$); this reduction in memory passes and operations makes it the standard choice in modern LLMs like LLaMA and Mistral.


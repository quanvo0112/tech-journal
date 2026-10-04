# Single Neuron

- **Problem Link:** [NeetCode - Single Neuron](https://neetcode.io/problems/single-neuron)
- **Difficulty:** `Easy`
- **NeetCode ML Module:** `Build a Neural Net`
- **Implementation Framework:** `NumPy`
- **Core Concept / Formulation:** `Artificial Neuron, Dot Product Pre-activation, Non-Linear Activation (Sigmoid & ReLU), Vectorized Forward Pass`
- **Last Practiced:** 2026-10-04
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

- **Objective / Problem Statement:**
  > Implement the forward inference pass of a single artificial neuron from scratch. Given a 1D input feature vector $x \in \mathbb{R}^n$, a 1D weight parameter vector $w \in \mathbb{R}^n$, a scalar bias $b \in \mathbb{R}$, and an activation function specifier (`activation \in \{"sigmoid", "relu"\}`), compute the pre-activation linear combination and apply the specified activation function. Return the final scalar output rounded to 5 decimal places.

- **1. Pre-activation (Affine Transformation):**
  $$z = x \cdot w + b = \sum_{i=1}^n x_i w_i + b$$

- **2. Non-linear Activation Mapping:**
  - **Sigmoid Activation:** Compresses real-valued inputs into the probability interval $(0, 1)$:
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
  - **Rectified Linear Unit (ReLU):** Thresholds negative activations at zero while acting as the identity for positive values:
    $$\text{ReLU}(z) = \max(0, z)$$

- **3. Output Rounding:**
  $$\hat{y} = \text{round}(a, 5) \quad \text{where } a = \begin{cases} \sigma(z), & \text{if activation} = \text{"sigmoid"} \\ \text{ReLU}(z), & \text{if activation} = \text{"relu"} \end{cases}$$

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Data Type | Description |
|:---:|:---:|:---:|:---:|:---|
| Input Features | $x$ | `(n,)` | `NDArray[np.float64]` | 1D feature vector of dimension $n$ |
| Weight Vector | $w$ | `(n,)` | `NDArray[np.float64]` | 1D learnable parameter weights |
| Bias | $b$ | `()` (scalar) | `float` | Scalar additive intercept parameter |
| Pre-activation | $z$ | `()` (scalar) | `float` / `np.float64` | Affine combination: $x \cdot w + b$ |
| Output | $\hat{y}$ | `()` (scalar) | `float` | Post-activation scalar rounded to 5 decimals |

---

## 3. Vectorized Implementation & Numerical Stability

1. **BLAS Dot Product Optimization:**
   - Instead of iterative looping $\sum_{i=0}^{n-1} x[i] \cdot w[i]$, `np.dot(x, w)` delegates directly to optimized low-level BLAS routines (`ddot` / `sdot`).
   - Computes the inner product in $\mathcal{O}(n)$ time with SIMD acceleration and zero Python loop overhead.

2. **Scalar Casting & Type Safety:**
   - `np.dot` on two 1D NumPy arrays returns a NumPy scalar (`np.float64`).
   - When evaluating `max(0.0, float(z))`, casting to native Python `float` prevents subtle type issues and ensures `round(float(output), 5)` strictly returns a Python `float` conforming to the expected return type annotation.

---

## 4. Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Where $n$ is the length of input array $x$ (and weight vector $w$).
  - Computing the dot product $x \cdot w$ requires $n$ multiplications and $n - 1$ additions.
  - Adding bias $b$, evaluating $\sigma(z)$ or $\max(0, z)$, and rounding take constant $\mathcal{O}(1)$ time.
- **Space Complexity:** $\mathcal{O}(1)$
  - The dot product and activation function operate directly on scalar intermediates with no auxiliary tensor allocations.

---

## 5. Edge Cases & Gotchas

- [x] **Negative Pre-activation with ReLU:** When $z \le 0$, `max(0.0, float(z))` correctly returns `0.0`.
- [x] **Zero Bias ($b = 0.0$):** Handled transparently by the linear sum $z = x \cdot w + 0.0$.
- [x] **Sigmoid Asymptotes:** For large positive $z$, $e^{-z} \to 0 \implies \sigma(z) \to 1.0$. For large negative $z$, $e^{-z} \to \infty \implies \sigma(z) \to 0.0$.
- [x] **1D Vector Guarantee:** Input $x$ and weights $w$ are 1D arrays of identical length $n$, ensuring `np.dot(x, w)` produces a scalar rather than a 2D matrix.

---

## 6. Clean Code (Optimal Implementation)

```python
import numpy as np
from numpy.typing import NDArray


class Solution:

    def forward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        activation: str
    ) -> float:
        # Pre-activation: affine linear combination
        z = np.dot(x, w) + b

        # Non-linear activation mapping
        if activation == "sigmoid":
            output = 1 / (1 + np.exp(-z))
        else:
            output = max(0.0, float(z))

        return round(float(output), 5)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### The Artificial Neuron Computation Flow

```text
Input Features x:  [x_1, x_2, ..., x_n]
Weights w:         [w_1, w_2, ..., w_n]
                         │
                         ▼
              Linear Combination (Affine)
                  z = np.dot(x, w) + b
                         │
                         ▼
               Activation Branching
           ┌─────────────┴─────────────┐
           │                           │
  activation == "sigmoid"      activation == "relu"
           │                           │
           ▼                           ▼
    σ(z) = 1 / (1 + e^-z)       ReLU(z) = max(0, z)
           │                           │
           └─────────────┬─────────────┘
                         │
                         ▼
                 Output Formatting
              round(float(output), 5)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** An artificial neuron is the foundational unit of neural computation, decomposing cleanly into an affine linear transformation ($z = x \cdot w + b$) followed by a non-linear activation function.

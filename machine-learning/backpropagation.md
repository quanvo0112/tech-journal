# Backpropagation

- **Problem Link:** [NeetCode - Backpropagation](https://neetcode.io/problems/backpropagation)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `Build a Neural Net`
- **Implementation Framework:** `NumPy`
- **Core Concept / Formulation:** `Backpropagation, Multivariate Chain Rule, Squared Error Loss Gradient, Sigmoid Derivative, Scalar-Vector Gradient Broadcasting`
- **Last Practiced:** 2026-10-04
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

- **Objective / Problem Statement:**
  > Implement the backward pass (backpropagation) for a single neuron equipped with sigmoid activation and half Mean Squared Error loss. Given input features $x \in \mathbb{R}^n$, weights $w \in \mathbb{R}^n$, scalar bias $b \in \mathbb{R}$, and scalar ground-truth target $y_{\text{true}} \in \mathbb{R}$, compute the forward activation and calculate the exact analytical gradients of the loss with respect to weights ($\frac{\partial \mathcal{L}}{\partial w}$) and bias ($\frac{\partial \mathcal{L}}{\partial b}$) via the Chain Rule. Return the weight gradients rounded to 5 decimal places (`np.round(..., 5)`) and the bias gradient rounded to 5 decimal places (`round(..., 5)`).

- **1. Forward Propagation Formulations:**
  1. **Pre-activation Affine Map:**
     $$z = x \cdot w + b = \sum_{i=1}^n x_i w_i + b$$
  2. **Sigmoid Non-Linear Activation:**
     $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
  3. **Loss Function (Half Squared Error):**
     $$\mathcal{L} = \frac{1}{2}(\hat{y} - y_{\text{true}})^2$$

- **2. Analytical Backward Pass (Chain Rule Decomposition):**
  We systematically backpropagate gradients from the objective loss backwards to the parameters:
  
  $$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$
  $$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial b}$$

  - **Step 1: Derivative of Loss w.r.t. Prediction ($\hat{y}$):**
    $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}} \left[ \frac{1}{2}(\hat{y} - y_{\text{true}})^2 \right] = \hat{y} - y_{\text{true}}$$

  - **Step 2: Derivative of Sigmoid w.r.t. Pre-activation ($z$):**
    $$\frac{\partial \hat{y}}{\partial z} = \frac{d}{dz} \left[ \frac{1}{1 + e^{-z}} \right] = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$

  - **Step 3: Pre-activation Error Signal ($\delta = \frac{\partial \mathcal{L}}{\partial z}$):**
    $$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = (\hat{y} - y_{\text{true}}) \cdot \hat{y}(1 - \hat{y})$$

  - **Step 4: Gradients w.r.t. Weights ($w$):**
    Since $\frac{\partial z}{\partial w} = x$:
    $$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z} \cdot x = (\hat{y} - y_{\text{true}})\hat{y}(1 - \hat{y}) \cdot x$$

  - **Step 5: Gradient w.r.t. Bias ($b$):**
    Since $\frac{\partial z}{\partial b} = 1$:
    $$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial z} \cdot 1 = \frac{\partial \mathcal{L}}{\partial z} = (\hat{y} - y_{\text{true}})\hat{y}(1 - \hat{y})$$

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Data Type | Description |
|:---:|:---:|:---:|:---:|:---|
| Input Features | $x$ | `(n,)` | `NDArray[np.float64]` | 1D feature vector of length $n$ |
| Weight Vector | $w$ | `(n,)` | `NDArray[np.float64]` | 1D learnable parameter weights |
| Bias | $b$ | `()` (scalar) | `float` | Scalar additive intercept parameter |
| Ground Truth | $y_{\text{true}}$ | `()` (scalar) | `float` | Scalar training target |
| Pre-activation | $z$ | `()` (scalar) | `float` / `np.float64` | Inner product $x \cdot w + b$ |
| Model Prediction | $\hat{y}$ | `()` (scalar) | `float` / `np.float64` | Sigmoid activation output |
| Pre-activation Grad | $\frac{\partial \mathcal{L}}{\partial z}$ | `()` (scalar) | `float` / `np.float64` | Intermediate error signal $\delta$ |
| Weight Gradient | $\frac{\partial \mathcal{L}}{\partial w}$ | `(n,)` | `NDArray[np.float64]` | Gradient vector w.r.t. weights $w$ |
| Bias Gradient | $\frac{\partial \mathcal{L}}{\partial b}$ | `()` (scalar) | `float` | Scalar gradient w.r.t. bias $b$ |

---

## 3. Vectorized Implementation & Numerical Stability

1. **Scalar-Vector Broadcasting:**
   - In NumPy, multiplying scalar error signal `dL_dz` with 1D input array `x` (`dL_dw = dL_dz * x`) performs vectorized scalar-vector element-wise multiplication in $\mathcal{O}(n)$ time without explicit iteration.
2. **Numerical Type Integrity:**
   - The returned tuple strictly requires `Tuple[NDArray[np.float64], float]`.
   - `np.round(dL_dw, 5)` returns an `NDArray[np.float64]`.
   - `round(float(dL_db), 5)` converts any intermediate NumPy float scalar back into a native Python `float`.

---

## 4. Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Where $n$ is the dimension of the feature vector $x$.
  - Forward pass computes `np.dot(x, w)` in $\mathcal{O}(n)$ time.
  - Computing scalar derivatives `dL_dyhat`, `dyhat_dz`, `dL_dz` takes $\mathcal{O}(1)$ time.
  - Computing weight gradients `dL_dz * x` scales an array of size $n$, taking $\mathcal{O}(n)$ time.
- **Space Complexity:** $\mathcal{O}(n)$
  - Storing and returning the gradient array `dL_dw` requires $\mathcal{O}(n)$ auxiliary memory.

---

## 5. Edge Cases & Gotchas

- [x] **Perfect Prediction ($\hat{y} == y_{\text{true}}$):** `dL_dyhat = 0.0`, resulting in zero gradient updates (`dL_dw = 0`, `dL_db = 0.0`).
- [x] **Sigmoid Vanishing Gradient:** If $|z|$ is large, $\hat{y} \to 0$ or $\hat{y} \to 1$. Consequently, `dyhat_dz = \hat{y}(1 - \hat{y}) \to 0$, causing gradients to vanish regardless of error size.
- [x] **Scalar vs 1D Array Shapes:** Ensure `dL_db` remains a scalar `float` rather than an array, while `dL_dw` preserves the original 1D shape `(n,)` of $w$.

---

## 6. Clean Code (Optimal Implementation)

```python
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:

    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float
    ) -> Tuple[NDArray[np.float64], float]:
        # 1. Forward Pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))

        # 2. Chain Rule Decomposition
        dL_dyhat = y_hat - y_true
        dyhat_dz = y_hat * (1 - y_hat)
        dL_dz = dL_dyhat * dyhat_dz

        # 3. Analytical Parameter Gradients
        dL_dw = dL_dz * x
        dL_db = dL_dz

        return np.round(dL_dw, 5), round(float(dL_db), 5)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Forward Propagation vs. Backpropagation Flow

```text
FORWARD PASS:
Inputs (x, w, b) ───► z = dot(x, w) + b ───► ŷ = σ(z) ───► Loss = 0.5 * (ŷ - y_true)^2

BACKWARD PASS (Backpropagation via Chain Rule):
Loss ───────────────► dL_dyhat = ŷ - y_true
                             │
                             ▼
                      dyhat_dz = ŷ * (1 - ŷ)
                             │
                             ▼
                      dL_dz = dL_dyhat * dyhat_dz
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
         dL_dw = dL_dz * x             dL_db = dL_dz
                │                         │
                ▼                         ▼
        np.round(dL_dw, 5)        round(float(dL_db), 5)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Backpropagation is the algorithmic application of the multivariate Chain Rule running in reverse: compute error signal $\frac{\partial \mathcal{L}}{\partial z}$ at the pre-activation junction, then distribute gradients to weights ($\delta \cdot x$) and bias ($\delta$).

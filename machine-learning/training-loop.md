# Training Loop

- **Problem Link:** [NeetCode - Training Loop](https://neetcode.io/problems/training-loop)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `Training`
- **Implementation Framework:** `NumPy`
- **Core Concept / Formulation:** `Iterative Training Loop, Batch Gradient Descent, Mean Squared Error (MSE) Optimization, Forward Pass, Analytical Gradients, Epoch Optimization`
- **Last Practiced:** 2026-10-08
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Objective
A **Training Loop** is the central algorithmic heartbeat of supervised machine learning. It orchestrates the cyclical interaction between model predictions (forward pass), loss evaluation, analytical gradient computation (backward pass), and parameter updates (optimizer step) across multiple epochs.

In this problem, we construct a complete batch training loop for **Linear Regression with learnable weights $w$ and bias $b$** from scratch using NumPy.

---

### Step-by-Step Derivation

Given a training dataset with $N$ samples and $M$ features:
- Feature matrix: $X \in \mathbb{R}^{N \times M}$
- Target vector: $y \in \mathbb{R}^N$
- Number of training iterations: $\text{epochs}$
- Learning rate: $\eta = \text{lr}$

#### 1. Initialization
Parameters are initialized to zero:
$$w = \mathbf{0} \in \mathbb{R}^M, \quad b = 0.0 \in \mathbb{R}$$

#### 2. Forward Pass (Linear Hypothesis)
For current parameters $w$ and $b$, the predicted outputs $\hat{y}$ are given by:
$$\hat{y} = Xw + b \in \mathbb{R}^N$$

#### 3. Loss Function (Mean Squared Error)
The Mean Squared Error (MSE) objective function measures average squared residual discrepancy:
$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 = \frac{1}{N} \|Xw + b\mathbf{1} - y\|_2^2$$

#### 4. Analytical Gradients via Vector Calculus
Let $\mathbf{e} = \hat{y} - y \in \mathbb{R}^N$ denote the residual error vector.

- **Gradient with respect to Weight Vector $w$:**
  Using the chain rule:
  $$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{N} \sum_{i=1}^N 2(\hat{y}_i - y_i) \frac{\partial \hat{y}_i}{\partial w} = \frac{2}{N} \sum_{i=1}^N e_i X_i^T = \frac{2}{N} X^T (\hat{y} - y) = \frac{2}{N} X^T \mathbf{e}$$
  - Shape: $(M \times N) \times (N \times 1) = (M \times 1) \equiv (M,)$.

- **Gradient with respect to Scalar Bias $b$:**
  $$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N 2(\hat{y}_i - y_i) \frac{\partial \hat{y}_i}{\partial b} = \frac{2}{N} \sum_{i=1}^N e_i (1) = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i) = \frac{2}{N} \sum \mathbf{e}$$
  - Shape: Scalar float.

#### 5. Gradient Descent Update
After computing both gradients for the full batch, update parameters simultaneously:
$$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}$$
$$b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input Features | $X$ | `(N, M)` | Training data matrix ($N$ samples, $M$ features) |
| Ground Truth | $y$ | `(N,)` | Target continuous labels |
| Weight Vector | $w$ | `(M,)` | Learnable feature weights |
| Bias Scalar | $b$ | `()` | Learnable additive offset |
| Predictions | $\hat{y}$ | `(N,)` | Model predictions: $Xw + b$ |
| Error Vector | $\mathbf{e}$ | `(N,)` | Discrepancy: $\hat{y} - y$ |
| Weight Gradient | $\nabla_w \mathcal{L}$ | `(M,)` | Vectorized gradient: $\frac{2}{N} X^T \mathbf{e}$ |
| Bias Gradient | $\nabla_b \mathcal{L}$ | `()` | Scalar gradient: $\frac{2}{N} \sum \mathbf{e}$ |

---

## 3. Vectorized Implementation & Numerical Stability

1. **Why Pure Vectorization over Explicit Feature Loops:**
   - Instead of iterating through each feature dimension $j \in [0, M-1]$:
     ```python
     # Inefficient O(M) Python loop
     for j in range(n_features):
         dw[j] = (2 / n_samples) * np.sum(error * X[:, j])
```
   - The matrix-vector dot product `X.T @ error` computes all $M$ gradient components in a single fused BLAS level-2 kernel, executing up to $50\times$ faster on modern CPUs.
2. **Synchronous Parameter Updates:**
   - Crucial algorithmic invariant: Both $dw$ and $db$ must be completely evaluated using the current epoch's state **before** modifying $w$ or $b$. Updating $w$ in-place while computing $db$ would corrupt the bias gradient.

---

## 4. Complexity Analysis

Let $N = \text{n\_samples}$, $M = \text{n\_features}$, and $E = \text{epochs}$:

- **Time Complexity:** $\mathcal{O}(E \cdot N \cdot M)$
  - In each epoch:
    - Forward prediction $X @ w$: requires $N \cdot M$ multiplications $\to \mathcal{O}(NM)$.
    - Residual error $\hat{y} - y$: element-wise subtraction $\to \mathcal{O}(N)$.
    - Weight gradient $X^T @ \text{error}$: requires $M \cdot N$ operations $\to \mathcal{O}(NM)$.
    - Bias gradient `np.sum(error)`: sum over $N$ items $\to \mathcal{O}(N)$.
    - Parameter update: $\mathcal{O}(M)$ for $w$ and $\mathcal{O}(1)$ for $b$.
  - Total time across all $E$ epochs: $\mathcal{O}(E \cdot NM)$.
- **Space Complexity:** $\mathcal{O}(N + M)$
  - Memory buffers allocated during execution:
    - Weight vector $w$: $\mathcal{O}(M)$.
    - Prediction $\hat{y}$ and residual error $\mathbf{e}$: $\mathcal{O}(N)$.
    - Gradient $dw$: $\mathcal{O}(M)$.
  - Total auxiliary space is strictly $\mathcal{O}(N + M)$.

---

## 5. Edge Cases & Gotchas

- [x] **Zero Initialization:** Starting with `w = np.zeros(n_features)` and `b = 0.0` ensures deterministic gradient descent paths. Unlike deep neural networks, convex linear regression surfaces have a single global minimum and do not require random initialization.
- [x] **Scale Factor $\frac{2}{N}$:** MSE is defined with $\frac{1}{N}$, so the power-rule derivative brings down $2$, resulting in the scalar factor $\frac{2}{N}$. Omitting the $2$ effectively halves the learning rate.
- [x] **Precision Rounding:** Return weights rounded with `np.round(w, 5)` and scalar bias with Python's `round(b, 5)`.

---

## 6. Clean Code

```python
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


class Solution:

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int,
        lr: float,
    ) -> Tuple[NDArray[np.float64], float]:
        n_samples, n_features = X.shape

        # 1. Initialize weights to zeros and bias to 0.0
        w = np.zeros(n_features)
        b = 0.0

        # 2. Execute the iterative training loop
        for _ in range(epochs):
            # Forward pass: compute linear predictions
            y_hat = X @ w + b

            # Discrepancy error
            error = y_hat - y

            # Analytical MSE gradients
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # Gradient descent parameter updates
            w -= lr * dw
            b -= lr * db

        # 3. Return rounded parameters (5 decimal places)
        return np.round(w, 5), round(b, 5)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### The Universal Training Loop Cycle

```text
                      Initialize w = 0, b = 0
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
       ┌───────────────────┐           ┌───────────────────┐
       │   FORWARD PASS    │           │   BACKWARD PASS   │
       │   y_hat = Xw + b  │ ───────>  │  error = y_hat - y│
       └───────────────────┘           │  dw = (2/N) X^T e │
                                       │  db = (2/N) sum(e)│
                                       └─────────┬─────────┘
                                                 │
                                                 ▼
                                       ┌───────────────────┐
                                       │   OPTIMIZER STEP  │
                                       │    w -= lr * dw   │
                                       │    b -= lr * db   │
                                       └─────────┬─────────┘
                                                 │
                                          Next Epoch / Exit
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** A training loop operationalizes the core cycle of learning: generate predictions, evaluate discrepancy errors, compute partial derivatives via vector calculus ($X^T \mathbf{e}$ and $\sum \mathbf{e}$), and step down the loss landscape via gradient descent.


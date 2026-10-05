# Multi-Layer Backpropagation

- **Problem Link:** [NeetCode - Multi-Layer Backpropagation](https://neetcode.io/problems/multi-layer-backpropagation)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `Build a Neural Net`
- **Implementation Framework:** `NumPy`
- **Core Concept / Formulation:** `2-Layer Multi-Layer Perceptron (MLP), Multivariate Matrix Calculus, ReLU Subgradient, Outer Product Weight Gradients, Vectorized Chain Rule`
- **Last Practiced:** 2026-10-05
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

- **Objective / Problem Statement:**
  > Implement both the complete forward pass and backward propagation pass for a 2-layer Neural Network (Multilayer Perceptron) with ReLU hidden activation and Mean Squared Error (MSE) loss. Given 1D input features $x \in \mathbb{R}^I$, hidden layer weights $W_1 \in \mathbb{R}^{H \times I}$ and bias $b_1 \in \mathbb{R}^H$, output layer weights $W_2 \in \mathbb{R}^{O \times H}$ and bias $b_2 \in \mathbb{R}^O$, and continuous targets $y_{\text{true}} \in \mathbb{R}^O$, compute model predictions $\hat{y}$, scalar loss $\mathcal{L}$, and exact analytical parameter gradients ($\frac{\partial \mathcal{L}}{\partial W_1}, \frac{\partial \mathcal{L}}{\partial b_1}, \frac{\partial \mathcal{L}}{\partial W_2}, \frac{\partial \mathcal{L}}{\partial b_2}$) rounded to 4 decimal places.

- **1. Forward Propagation Formulations:**
  1. **Hidden Layer Pre-activation ($z_1$):**
     $$z_1 = W_1 x + b_1 \in \mathbb{R}^H$$
  2. **Hidden Layer Activation ($a_1$ via ReLU):**
     $$a_1 = \text{ReLU}(z_1) = \max(0, z_1) \in \mathbb{R}^H$$
  3. **Output Layer Predictions ($\hat{y}$):**
     $$\hat{y} = W_2 a_1 + b_2 \in \mathbb{R}^O$$
  4. **Mean Squared Error (MSE) Objective:**
     $$\mathcal{L} = \frac{1}{O} \sum_{k=1}^O (\hat{y}_k - y_k)^2$$

- **2. Analytical Backward Propagation (Chain Rule Breakdown):**
  We propagate gradients backward from the scalar loss through each intermediate operation:

  - **Step 1: Output Prediction Error Gradient ($\frac{\partial \mathcal{L}}{\partial \hat{y}}$):**
    $$\delta_{\text{out}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{2}{O} (\hat{y} - y_{\text{true}}) \in \mathbb{R}^O$$

  - **Step 2: Output Layer Parameter Gradients ($W_2, b_2$):**
    Since $\hat{y} = W_2 a_1 + b_2$:
    $$\frac{\partial \mathcal{L}}{\partial W_2} = \delta_{\text{out}} a_1^T = \text{np.outer}(\delta_{\text{out}}, a_1) \in \mathbb{R}^{O \times H}$$
    $$\frac{\partial \mathcal{L}}{\partial b_2} = \delta_{\text{out}} \in \mathbb{R}^O$$

  - **Step 3: Backpropagating Error to Hidden Activations ($a_1$):**
    By the multivariate chain rule across linear projection $W_2$:
    $$\frac{\partial \mathcal{L}}{\partial a_1} = W_2^T \delta_{\text{out}} = W_2^T \frac{\partial \mathcal{L}}{\partial \hat{y}} \in \mathbb{R}^H$$

  - **Step 4: Backpropagating Error Through ReLU Activation ($z_1$):**
    The derivative of ReLU is the unit step function:
    $$\frac{d \text{ReLU}(z)}{dz} = \begin{cases} 1, & z > 0 \\ 0, & z \le 0 \end{cases}$$
    Applying the Hadamard (element-wise) product:
    $$\delta_1 = \frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial a_1} \odot \mathbb{I}(z_1 > 0) \in \mathbb{R}^H$$

  - **Step 5: Hidden Layer Parameter Gradients ($W_1, b_1$):**
    Since $z_1 = W_1 x + b_1$:
    $$\frac{\partial \mathcal{L}}{\partial W_1} = \delta_1 x^T = \text{np.outer}(\delta_1, x) \in \mathbb{R}^{H \times I}$$
    $$\frac{\partial \mathcal{L}}{\partial b_1} = \delta_1 \in \mathbb{R}^H$$

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Data Type | Description |
|:---:|:---:|:---:|:---:|:---|
| Input Features | $x$ | `(I,)` | `NDArray[np.float64]` | 1D feature vector of dimension $I$ |
| Hidden Weights | $W_1$ | `(H, I)` | `NDArray[np.float64]` | Layer 1 weight projection matrix |
| Hidden Bias | $b_1$ | `(H,)` | `NDArray[np.float64]` | Layer 1 additive bias vector |
| Hidden Pre-activation | $z_1$ | `(H,)` | `NDArray[np.float64]` | Affine combination: $W_1 x + b_1$ |
| Hidden Activation | $a_1$ | `(H,)` | `NDArray[np.float64]` | Post-ReLU activation: $\max(0, z_1)$ |
| Output Weights | $W_2$ | `(O, H)` | `NDArray[np.float64]` | Layer 2 weight projection matrix |
| Output Bias | $b_2$ | `(O,)` | `NDArray[np.float64]` | Layer 2 additive bias vector |
| Predictions | $\hat{y}$ | `(O,)` | `NDArray[np.float64]` | Output layer prediction vector |
| Ground Truth | $y_{\text{true}}$ | `(O,)` | `NDArray[np.float64]` | Target continuous values |
| Loss | $\mathcal{L}$ | `()` (scalar) | `float` | Scalar Mean Squared Error |
| Output Weight Gradient | $\frac{\partial \mathcal{L}}{\partial W_2}$ | `(O, H)` | `NDArray[np.float64]` | Outer product $\delta_{\text{out}} \otimes a_1$ |
| Output Bias Gradient | $\frac{\partial \mathcal{L}}{\partial b_2}$ | `(O,)` | `NDArray[np.float64]` | Prediction error signal $\delta_{\text{out}}$ |
| Hidden Weight Gradient | $\frac{\partial \mathcal{L}}{\partial W_1}$ | `(H, I)` | `NDArray[np.float64]` | Outer product $\delta_1 \otimes x$ |
| Hidden Bias Gradient | $\frac{\partial \mathcal{L}}{\partial b_1}$ | `(H,)` | `NDArray[np.float64]` | Pre-activation error signal $\delta_1$ |

---

## 3. Vectorized Implementation & Numerical Stability

1. **Matrix Multiplication vs. Outer Products:**
   - Pre-activations use standard matrix-vector dot products via the `@` operator (`W1 @ x + b1`, `W2 @ a1 + b2`).
   - Parameter weight gradients represent the outer product of error vectors with incoming activation vectors. `np.outer(u, v)` yields the exact rank-1 update matrix of dimension `(len(u), len(v))` in $\mathcal{O}(\text{len}(u) \cdot \text{len}(v))$ time without explicit tensor reshaping.
2. **ReLU Subgradient Masking:**
   - In NumPy, boolean masking `(z1 > 0)` casts directly to integers/floats during arithmetic multiplication (`da1 * (z1 > 0)`), zeroing out gradients for all inactive neurons in a single vectorized instruction.
3. **Output Formatting & Precision:**
   - Scalar `loss` rounded via `round(float(loss), 4)`.
   - Gradient matrices and vectors rounded via `np.round(..., 4).tolist()` to return native nested Python lists as specified by the problem template.

---

## 4. Complexity Analysis

- **Time Complexity:** $\mathcal{O}(H \cdot I + O \cdot H)$
  - **Forward Pass:** Computing $W_1 x$ takes $\mathcal{O}(H \cdot I)$ operations; computing $W_2 a_1$ takes $\mathcal{O}(O \cdot H)$ operations. ReLU and MSE are $\mathcal{O}(H)$ and $\mathcal{O}(O)$.
  - **Backward Pass:** Transposed projection $W_2^T \delta_{\text{out}}$ takes $\mathcal{O}(O \cdot H)$. Outer products $\text{np.outer}(\delta_{\text{out}}, a_1)$ and $\text{np.outer}(\delta_1, x)$ take $\mathcal{O}(O \cdot H)$ and $\mathcal{O}(H \cdot I)$ operations respectively.
  - Overall time complexity is linear with respect to the total number of network parameters.
- **Space Complexity:** $\mathcal{O}(H \cdot I + O \cdot H)$
  - Required to store cached forward activations ($z_1, a_1$) and construct gradient matrices ($dW_1, dW_2$).

---

## 5. Edge Cases & Gotchas

- [x] **Dying ReLU Inactive Units:** If $z_{1, j} \le 0$, `(z1 > 0)[j] == False`, entirely cutting off the gradient flow through hidden unit $j$. Consequently, row $j$ of $dW_1$ and entry $j$ of $db_1$ receive strictly zero gradient.
- [x] **MSE Normalization Denominator:** The derivative of $\frac{1}{O} \sum (\hat{y}_k - y_k)^2$ is $\frac{2}{O}(\hat{y} - y)$, where $O = \text{len}(y_{\text{true}})$. Omitting the division by $O$ scales gradients by an erroneous constant factor.
- [x] **Transposed Weight Projection in Backprop:** When propagating error backward from layer 2 to hidden activations, the incoming gradient must be pre-multiplied by $W_2^T$ (`W2.T @ dL_dpred`), matching the adjoint mapping of the forward transformation $W_2 a_1$.

---

## 6. Clean Code (Optimal Implementation)

```python
import numpy as np
from typing import List


class Solution:

    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:
        # Convert all inputs to NumPy arrays with float precision
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # ----------------- FORWARD PASS -----------------
        # Layer 1: Affine projection + ReLU activation
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)

        # Layer 2: Affine projection to output
        predictions = W2 @ a1 + b2

        # Objective: Mean Squared Error loss
        loss = np.mean((predictions - y_true) ** 2)

        # ----------------- BACKWARD PASS -----------------
        # Output layer error: dL / d(pred) = (2 / O) * (pred - y_true)
        dL_dpred = 2 * (predictions - y_true) / len(y_true)

        # Output layer gradients
        dW2 = np.outer(dL_dpred, a1)
        db2 = dL_dpred

        # Backpropagation to hidden activations: dL / da1 = W2^T * dL_dpred
        da1 = W2.T @ dL_dpred

        # Backpropagation through ReLU: dL / dz1 = da1 * I(z1 > 0)
        dz1 = da1 * (z1 > 0)

        # Hidden layer gradients
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Two-Layer Forward and Backward Propagation Flow

```text
FORWARD PROPAGATION:
Input x (I,)
    │
    ▼
[ W1 @ x + b1 ] ────────► z1 (H,) ───► a1 = max(0, z1)
                                            │
                                            ▼
                                  [ W2 @ a1 + b2 ] ───► ŷ (O,) ───► Loss = MSE(ŷ, y_true)

BACKWARD PROPAGATION:
Loss ──────────► dL_dpred = (2 / O) * (ŷ - y_true)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 dW2 = outer(dL_dpred, a1)     db2 = dL_dpred
        │
        ▼
da1 = W2.T @ dL_dpred
        │
        ▼
dz1 = da1 * (z1 > 0)           <─── ReLU Gradient Mask
        │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 dW1 = outer(dz1, x)           db1 = dz1
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Multi-layer backpropagation is the recursive application of the Chain Rule: error gradients are projected backward through transposed weight matrices ($W_{l+1}^T \delta$), modulated by local activation derivatives ($\sigma'(z_l)$), and paired with forward activations via outer products to yield weight gradients ($\delta_l \otimes a_{l-1}$).

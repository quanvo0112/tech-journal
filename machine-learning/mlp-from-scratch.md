# MLP From Scratch

- **Problem Link:** [NeetCode - MLP From Scratch](https://neetcode.io/problems/mlp-from-scratch)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `Build a Neural Net`
- **Implementation Framework:** `NumPy`
- **Core Concept / Formulation:** `Multi-Layer Perceptron (MLP), Forward Inference Pipeline, Row-Vector Affine Transformation (x @ W + b), Hidden ReLU Non-Linearity, Linear Output Layer`
- **Last Practiced:** 2026-10-05
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

- **Objective / Problem Statement:**
  > Implement the generalized forward inference pass of an arbitrary $L$-layer Multi-Layer Perceptron (MLP) from scratch using NumPy. Given an input feature vector $x \in \mathbb{R}^{D_0}$, a list of weight projection matrices $[W_0, W_1, \dots, W_{L-1}]$, and a list of additive bias vectors $[b_0, b_1, \dots, b_{L-1}]$, sequentially propagate activations through each fully-connected layer. Apply the Rectified Linear Unit ($\text{ReLU}$) non-linearity after every hidden layer ($l < L - 1$), while leaving the final output layer ($l = L - 1$) as a pure linear projection. Return the final output activations rounded to 5 decimal places.

- **1. Matrix Orientation & Affine Convention:**
  In classical mathematical literature, vectors are often column vectors ($W x + b$). However, modern deep learning frameworks (PyTorch, TensorFlow, and NeetCode ML) standardly represent weight matrices under the **row-vector convention**:
  $$W_l \in \mathbb{R}^{D_l \times D_{l+1}}, \quad b_l \in \mathbb{R}^{D_{l+1}}$$
  where $D_l$ denotes the input dimension to layer $l$ and $D_{l+1}$ denotes its output dimension.
  The affine transformation for an activation vector $a_{l-1} \in \mathbb{R}^{D_l}$ is therefore formulated as:
  $$z_l = a_{l-1} W_l + b_l \in \mathbb{R}^{D_{l+1}} \quad \iff \quad \text{output} = \text{output} \mathbin{@} W_l + b_l$$

- **2. Layer-by-Layer Forward Recurrence:**
  - **Input Layer:**
    $$a_{-1} = x \in \mathbb{R}^{D_0}$$
  - **Hidden Layers ($l = 0, 1, \dots, L - 2$):**
    $$z_l = a_{l-1} W_l + b_l$$
    $$a_l = \text{ReLU}(z_l) = \max(0, z_l)$$
  - **Output Layer ($l = L - 1$):**
    $$z_{L-1} = a_{L-2} W_{L-1} + b_{L-1}$$
    $$\hat{y} = z_{L-1} \quad (\text{No activation applied})$$
  - **Output Formatting:**
    $$\text{return } \text{np.round}(\hat{y}, 5)$$

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Data Type | Description |
|:---:|:---:|:---:|:---:|:---|
| Input Features | $x$ | `(D_0,)` | `NDArray[np.float64]` | Initial 1D feature vector |
| Layer $l$ Weights | $W_l$ | `(D_l, D_{l+1})` | `NDArray[np.float64]` | Projection matrix from $D_l$ to $D_{l+1}$ |
| Layer $l$ Bias | $b_l$ | `(D_{l+1},)` | `NDArray[np.float64]` | Additive bias vector for layer $l$ |
| Pre-activation | $z_l$ | `(D_{l+1},)` | `NDArray[np.float64]` | Affine result: $a_{l-1} W_l + b_l$ |
| Hidden Activation | $a_l$ | `(D_{l+1},)` | `NDArray[np.float64]` | Post-ReLU activation: $\max(0, z_l)$ |
| Final Output | $\hat{y}$ | `(D_L,)` | `NDArray[np.float64]` | Final unconstrained prediction vector |

---

## 3. Vectorized Implementation & Numerical Stability

1. **Row-Vector Dot Product via `@` Operator:**
   - In NumPy, multiplying a 1D array of shape `(D_l,)` with a 2D matrix of shape `(D_l, D_{l+1})` using the `@` operator performs efficient BLAS vector-matrix multiplication (`gemv`), directly returning a 1D array of shape `(D_{l+1},)`.
   - Adding 1D bias vector `b[i]` of shape `(D_{l+1},)` performs element-wise vector addition in $\mathcal{O}(D_{l+1})$ time without broadcasting ambiguities.
2. **In-Place Activation Reuse:**
   - Rather than storing all intermediate activations in a list, inference sequentially updates a single variable `output = output @ weights[i] + biases[i]`.
   - This keeps peak auxiliary memory bounded strictly by $\mathcal{O}(\max_l D_l)$.
3. **Element-Wise ReLU via `np.maximum`:**
   - `np.maximum(0, output)` evaluates $\max(0, z_j)$ element-wise using SIMD instructions, which is vastly faster than Python list comprehensions.

---

## 4. Complexity Analysis

- **Time Complexity:** $\mathcal{O}\left( \sum_{l=0}^{L-1} D_l \cdot D_{l+1} \right)$
  - Where $L = \text{len}(\text{weights})$ is the total number of layers.
  - At each layer $l$, the vector-matrix product requires $D_l \cdot D_{l+1}$ multiplications and additions.
  - Evaluating ReLU and bias addition takes $\mathcal{O}(D_{l+1})$ operations.
  - Total compute time is linear with respect to the total number of model parameters (weights and biases).
- **Space Complexity:** $\mathcal{O}(D_{\max})$
  - Where $D_{\max} = \max(D_0, D_1, \dots, D_L)$ is the maximum layer dimension.
  - The iterative loop overwrites the intermediate activation array in-place, requiring memory proportional only to the largest layer's activation buffer.

---

## 5. Edge Cases & Gotchas

- [x] **Weight Matrix Orientation (Crucial Gotcha):**
  - **Common Mistake:** Assuming the classical textbook orientation $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$ and computing `weights[i] @ output` causes shape mismatch errors (e.g., `(2, 1) @ (2,) -> shape error`).
  - **Correct Invariant:** NeetCode ML formats weights as `(input_dim, output_dim)`. The valid multiplication order is strictly `output @ weights[i]`.
- [x] **Skipping Activation on the Final Layer:**
  - If ReLU is mistakenly applied to the output layer, all negative target values/logits are clamped to zero. The conditional check `if i < len(weights) - 1:` guarantees the final layer remains a linear transformation.
- [x] **Single-Layer MLP ($L = 1$):**
  - For a network with only one layer, the condition `i < len(weights) - 1` evaluates to `0 < 0` (False). ReLU is bypassed and the raw affine projection is returned directly.

---

## 6. Clean Code (Optimal Implementation)

```python
import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:

    def forward(
        self,
        x: NDArray[np.float64],
        weights: List[NDArray[np.float64]],
        biases: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        output = x

        for i in range(len(weights)):
            # Affine transformation: output @ W + b
            output = output @ weights[i] + biases[i]

            # Apply ReLU to all hidden layers, but omit on output layer
            if i < len(weights) - 1:
                output = np.maximum(0, output)

        return np.round(output, 5)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Generalized Multi-Layer Perceptron Feed-Forward Pipeline

```text
Input Features x (D_0,)
         │
         ▼
┌─────────────────────────────────┐
│ Layer 0: Hidden Layer           │
│   output = output @ W_0 + b_0   │  (D_0,) @ (D_0, D_1) -> (D_1,)
│   output = np.maximum(0, output)│  ReLU Non-linearity
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Layer 1: Hidden Layer           │
│   output = output @ W_1 + b_1   │  (D_1,) @ (D_1, D_2) -> (D_2,)
│   output = np.maximum(0, output)│  ReLU Non-linearity
└─────────────────────────────────┘
         │
         ▼
        ...
         │
         ▼
┌─────────────────────────────────┐
│ Layer L-1: Output Layer         │
│   output = output @ W_L-1 + b_L-1 (D_L-1,) @ (D_L-1, D_L) -> (D_L,)
│   (No ReLU activation applied)  │  Pure Linear Projection
└─────────────────────────────────┘
         │
         ▼
     np.round(output, 5)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** In feed-forward networks, always verify weight orientation: when weights are shaped `(D_in, D_out)`, perform row-vector multiplication `output @ W + b`, and strictly restrict activation non-linearities to hidden layers.

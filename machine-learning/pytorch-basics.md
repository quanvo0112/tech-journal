# PyTorch Basics

- **Problem Link:** [NeetCode - PyTorch Basics](https://neetcode.io/problems/pytorch-basics)
- **Difficulty:** `Easy`
- **NeetCode ML Module:** `PyTorch`
- **Implementation Framework:** `PyTorch`
- **Core Concept / Formulation:** `Tensor Reshaping, Column-Wise Mean Reduction, Feature Concatenation (dim=1), Mean Squared Error Loss`
- **Last Practiced:** 2026-10-06
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Objective
Modern deep learning frameworks rely on multi-dimensional tensors with automatic differentiation and hardware acceleration. Transitioning from NumPy to PyTorch requires mastering standard tensor primitives: structural reshaping, dimensional reductions, tensor joining, and canonical loss evaluations.

---

### 1. Tensor Reshaping (`reshape`)
Given an input matrix $X \in \mathbb{R}^{M \times N}$ containing a total of $K = M \cdot N$ scalar elements, we want to reconfigure its dimensional geometry into:
$$X_{\text{reshaped}} \in \mathbb{R}^{\frac{M \cdot N}{2} \times 2}$$

- **Preservation Invariant:** The total number of elements must remain invariant:
  $$(M \cdot N // 2) \times 2 = M \cdot N$$
- In PyTorch, `torch.reshape(input, shape)` returns a contiguous view without copying memory whenever possible. If the underlying tensor is non-contiguous in memory, it automatically returns a cloned copy with the target stride layout.

---

### 2. Column-Wise Mean Reduction (`average`)
Given a 2D tensor $X \in \mathbb{R}^{M \times N}$, the column-wise mean computes the expected value of each feature across all batch samples (rows):
$$\mu_j = \frac{1}{M} \sum_{i=0}^{M - 1} X_{i, j}, \quad \forall j \in \{0, 1, \dots, N - 1\}$$

- **Reduction Dimension:** Collapsing across rows corresponds to `dim=0`. The output shape drops from $(M, N)$ to a 1D tensor $(N,)$.

---

### 3. Feature-Wise Concatenation (`concatenate`)
Given two tensors $A \in \mathbb{R}^{M \times N_1}$ and $B \in \mathbb{R}^{M \times N_2}$ sharing identical row counts $M$, join them side-by-side along the column dimension:
$$C = [A \mid B] \in \mathbb{R}^{M \times (N_1 + N_2)}$$

- **Dimensional Invariant:** Joining side-by-side requires concatenating along `dim=1`. All dimensions other than the concatenation axis must strictly match ($A.\text{shape}[0] == B.\text{shape}[0]$).

---

### 4. Mean Squared Error Loss (`get_loss`)
Given a model prediction tensor $\hat{y} \in \mathbb{R}^n$ and ground-truth target tensor $y \in \mathbb{R}^n$, the Mean Squared Error (MSE) loss quantifies the average squared residual:
$$\mathcal{L}_{\text{MSE}}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2$$

- PyTorch provides optimized C++/CUDA implementations via `torch.nn.functional.mse_loss(prediction, target)`.

---

## 2. Tensor Shapes & Dimension Architecture

| Operation | Input Tensor(s) | Output Tensor | Formula / API Call |
| :--- | :---: | :---: | :--- |
| **Reshape** | $(M, N)$ | $(M \cdot N // 2, 2)$ | `torch.reshape(to_reshape, (M * N // 2, 2))` |
| **Column Mean** | $(M, N)$ | $(N,)$ | `torch.mean(to_avg, dim=0)` |
| **Concatenation** | $(M, N_1), (M, N_2)$ | $(M, N_1 + N_2)$ | `torch.cat((cat_one, cat_two), dim=1)` |
| **MSE Loss** | $(n,), (n,)$ | Scalar $()$ | `torch.nn.functional.mse_loss(prediction, target)` |

---

## 3. Implementation Invariants & Key Concepts

1. **The Semantic Meaning of `dim` in Reductions:**
   - In PyTorch, specifying `dim=k` means: **"eliminate axis $k$ by aggregating across it."**
   - Setting `dim=0` aggregates across rows (yielding column-wise statistics).
   - Setting `dim=1` aggregates across columns (yielding row-wise statistics).

2. **Memory Layout (`reshape` vs `view`):**
   - `tensor.view()` strictly requires the tensor to be contiguous in memory (`tensor.is_contiguous() == True`). If strides are non-standard (e.g., after `transpose()` or `permute()`), `view()` throws a runtime error.
   - `torch.reshape()` is safer: it returns a view if memory layout permits, but seamlessly falls back to copying if non-contiguous.

3. **Concatenation Dimension Alignment:**
   - For `torch.cat((a, b), dim=1)`, the batch/row dimension `shape[0]` must be identical. If rows do not match, PyTorch raises `RuntimeError: Sizes of tensors must match except in dimension 1`.

---

## 4. Complexity Analysis

Let $K$ denote the total number of scalar elements in the input tensor ($K = M \cdot N$):

- **`reshape`:**
  - **Time Complexity:** $\mathcal{O}(1)$ amortized when returning a view of contiguous memory.
  - **Space Complexity:** $\mathcal{O}(1)$ auxiliary space without allocating new memory buffers.
- **`average`:**
  - **Time Complexity:** $\mathcal{O}(K)$ to traverse and sum all elements along axis 0.
  - **Space Complexity:** $\mathcal{O}(N)$ for the returned 1D column-mean tensor.
- **`concatenate`:**
  - **Time Complexity:** $\mathcal{O}(M \cdot (N_1 + N_2))$ to copy elements into the newly allocated contiguous block.
  - **Space Complexity:** $\mathcal{O}(M \cdot (N_1 + N_2))$ to materialize the combined tensor.
- **`get_loss`:**
  - **Time Complexity:** $\mathcal{O}(n)$ vectorized element-wise subtraction, squaring, and mean reduction.
  - **Space Complexity:** $\mathcal{O}(1)$ auxiliary space returning a scalar tensor.

---

## 5. Edge Cases & Gotchas

- [x] **Integer Division in Shape Tuple:** Use integer floor division `M * N // 2` rather than float division `M * N / 2` to ensure dimension arguments are pure integers.
- [x] **Reduction Axis Selection:** Forgetting `dim=0` in `torch.mean` would reduce the entire tensor to a single scalar mean rather than preserving column feature statistics.
- [x] **Matching Non-Concatenation Dimensions:** When joining side-by-side with `dim=1`, ensure both input tensors have identical outer batch dimensions.
- [x] **Tensor Typing & Precision:** `F.mse_loss` expects both prediction and target tensors to share the same `dtype` (e.g., `torch.float32`), automatically broadcasting compatible shapes.

---

## 6. Clean Code

```python
import torch
import torch.nn
from torchtyping import TensorType

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        # Use torch.reshape(tensor, new_shape)
        M, N = to_reshape.shape
        return torch.reshape(to_reshape, (M * N // 2, 2))

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        # Use torch.mean(tensor, dim=0)
        return torch.mean(to_avg, dim=0)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        return torch.cat((cat_one, cat_two), dim=1)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        return torch.nn.functional.mse_loss(prediction, target)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Visual Transformation Map

```text
1. Reshape:
   ┌───────────┐           ┌───────┐
   │ M rows    │    ───>   │ M*N/2 │
   │ N columns │           │ 2 cols│
   └───────────┘           └───────┘

2. Column Average (dim=0):
   ┌───────────┐
   │ row 0     │
   │ row 1     │    ───>   [mean_col0, mean_col1, ..., mean_colN-1]
   │ row M-1   │
   └───────────┘

3. Concatenate (dim=1):
   ┌───────┐   ┌───────┐           ┌───────────────┐
   │   A   │ + │   B   │    ───>   │   A   │   B   │
   │(M, N1)│   │(M, N2)│           │ (M,  N1 + N2) │
   └───────┘   └───────┘           └───────────────┘

4. Mean Squared Error:
   predictions  ───┐
                   ├───>  (pred - target)^2  ───>  mean()  ───>  Scalar Loss
   targets      ───┘
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Mastering `reshape`, `mean(dim=0)`, `cat(..., dim=1)`, and `F.mse_loss` establishes the foundational tensor manipulation pipeline required for constructing, training, and diagnosing PyTorch neural architectures.

# Word Embeddings

- **Problem Link:** [NeetCode - Word Embeddings](https://neetcode.io/problems/word-embeddings)
- **Difficulty:** `Easy`
- **NeetCode ML Module:** `NLP`
- **Implementation Framework:** `NumPy`
- **Core Concept / Formulation:** `Embedding Lookup Table, Advanced Integer Indexing, Dense Vector Representation`
- **Last Practiced:** 2026-10-12
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)

---

## 1. Mathematical Formulation & Derivations

### Overview & Objective
Word embeddings are dense, low-dimensional, learnable vector representations of tokens (words or subwords). Instead of treating tokens as sparse one-hot vectors, an embedding matrix $E$ maps each discrete token ID to a continuous vector in $\mathbb{R}^d$.

The lookup operation retrieves the embedding vectors for a sequence of token IDs. Given an embedding matrix $E \in \mathbb{R}^{V \times d}$ (where $V$ is the vocabulary size and $d$ is the embedding dimension) and a sequence of token IDs $\text{ids} \in \mathbb{Z}^{k}$, the output is:
$$\text{Output} = E[\text{ids}] \in \mathbb{R}^{k \times d}$$

### Key Equations
Each row of the output corresponds to the embedding of one token:
$$Y_{j,:} = E_{\text{ids}_j, :}, \quad \forall j \in \{0, \dots, k-1\}$$

The lookup is a pure gather operation — no arithmetic is performed, only row selection by index. Duplicate IDs produce duplicate rows, and the original order is preserved.

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Embedding Matrix | $E$ | `(V, d)` | Learned lookup table; each row is a token's vector |
| Token IDs | $\text{ids}$ | `(k,)` | 1D array of integer token IDs |
| Output Vectors | $Y$ | `(k, d)` | Gathered embedding vectors for each token |

Where $V$ = vocabulary size, $d$ = embedding dimension, $k$ = number of tokens to look up.

---

## 3. Vectorized Implementation & Numerical Stability

1. **NumPy Advanced (Fancy) Indexing:**
   - `embeddings[token_ids]` performs a single vectorized gather. Internally, NumPy stacks the selected rows into a new contiguous array of shape `(k, d)`.
   - This avoids an explicit Python loop (`for i, idx in enumerate(token_ids): out[i] = embeddings[idx]`), which would incur interpreter overhead and per-row allocations.

2. **Semantic Guarantees of Fancy Indexing:**
   - **Order-preserving:** output row $j$ corresponds to input ID `token_ids[j]`.
   - **Duplicate-preserving:** if an ID appears twice, its embedding appears twice.
   - **Out-of-bounds safety:** NumPy raises `IndexError` for IDs outside $[0, V-1]$, guarding against silent corruption.

3. **Rounding Invariant:**
   - Per problem specifications, the returned array is rounded element-wise to 5 decimal places via `np.round(..., 5)`.

---

## 4. Complexity Analysis

Let $k$ = number of token IDs, $d$ = embedding dimension:

- **Time Complexity:** $O(k \cdot d)$
  - Each of the $k$ lookups copies $d$ floating-point values. The gather touches exactly $k \cdot d$ elements; no search over the vocabulary is needed since IDs are direct row indices.
- **Space Complexity:** $O(k \cdot d)$
  - The returned array materializes $k$ vectors of length $d$. No auxiliary structures beyond the output are required.

---

## 5. Edge Cases & Gotchas

- [x] **Repeated Token IDs:** Advanced indexing preserves duplicates; `lookup(E, [3, 3, 3])` returns three identical rows.
- [x] **Order Preservation:** Output follows the input sequence order, not sorted ID order — critical for downstream sequence models.
- [x] **Out-of-Bounds IDs:** IDs must satisfy $\text{ids}_j \in [0, V-1]$; otherwise NumPy raises `IndexError`.
- [x] **Empty Sequence:** An empty `token_ids` yields an empty result of shape `(0, d)`.
- [x] **Rounding Precision:** Round to **5 decimal places** (`np.round(..., 5)`) as specified by the problem template.
- [x] **Copy vs View:** Returns a new array (copy); mutating the output does not affect the embedding matrix.

---

## 6. Clean Code

```python
import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        return np.round(embeddings[token_ids], 5)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Visual Lookup Example

```text
Embedding Matrix E (V=4, d=3):
             col0    col1    col2
   row 0  [  0.10,  0.20,  0.30 ]
   row 1  [  0.40,  0.50,  0.60 ]
   row 2  [  0.70,  0.80,  0.90 ]
   row 3  [  1.00,  1.10,  1.20 ]

token_ids = [2, 0, 3]

Gather rows 2, 0, 3 → output shape (3, 3):
   [ 0.70, 0.80, 0.90 ]   ← E[2]
   [ 0.10, 0.20, 0.30 ]   ← E[0]
   [ 1.00, 1.10, 1.20 ]   ← E[3]

Note: order preserved (follows token_ids), duplicates would repeat rows.
```

---

### Architectural Pipeline

```text
   Token IDs: ids ∈ Z^k
            ↓
   Embedding Table: E ∈ R^(V × d)
            ↓
   Advanced Indexing: E[ids]   (vectorized row gather)
            ↓
   Output Matrix: Y ∈ R^(k × d)
            ↓
   Element-wise Round: np.round(Y, 5)
            ↓
   Dense Token Representations
```

* **Next Review Date:** 
* **Key Takeaway:** Embedding lookup is nothing more than row selection from a learnable matrix $E$ — NumPy's advanced indexing `E[token_ids]` performs it in a single vectorized gather with $O(kd)$ time and space, preserving both order and duplicates.

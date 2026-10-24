# Multi-Headed Self Attention

- **Problem Link:** [NeetCode - Multi-Headed Self Attention](https://neetcode.io/problems/multi-headed-self-attention)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `Attention & Transformers`
- **Implementation Framework:** `PyTorch`
- **Core Concept / Formulation:** `Multi-Head Attention, Parallel Attention Subspaces, Head Concatenation, Output Projection, Causal Masking`
- **Last Practiced:** 2026-10-24
- **Proficiency Level:**
  - [ ] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Motivation
A single attention head computes one set of attention weights per token, forcing all relationships (syntactic, semantic, positional) into a single distribution. **Multi-head attention** runs $H$ attention operations in parallel, each in its own lower-dimensional subspace, then merges them:

$$
\text{MHA}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_H)\, W_O
$$

Each head has its own projection matrices and its own causal attention:

$$
\text{head}_i = \operatorname{Attention}\!\left(XW_Q^{(i)},\; XW_K^{(i)},\; XW_V^{(i)}\right)
$$

$$
\operatorname{Attention}(Q_i, K_i, V_i) =
\operatorname{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_h}} + M\right) V_i
$$

with $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{E \times d_h}$ and $d_h = A / H$, where:

- $E$ = `embedding_dim`
- $A$ = `attention_dim` (total attention width)
- $H$ = `num_heads`
- $d_h$ = `head_size`

The concatenation of all heads has width $H \cdot d_h = A$, so the output projection $W_O \in \mathbb{R}^{A \times A}$ maps back to the full attention width. Splitting $A$ across heads keeps the total parameter count roughly the same as one full-width head, while letting each head specialize.

### Why Multiple Heads?
- **Different representation subspaces:** each head can learn a different notion of relevance (e.g. one tracks adjacent tokens, another tracks long-range dependencies). Averaging them into a single head would blur these signals.
- **Parallel, not sequential:** all heads consume the same $X$ and run independently, so the computation is still fully vectorizable across the batch.
- **Information mixing:** concatenation only places the heads side by side; $W_O$ is what actually combines information *across* heads. Without it, the heads would remain independent blocks.

### Causal Mask
The mask is identical for every head:

$$
M_{ts} =
\begin{cases}
0, & s \le t \\
-\infty, & s > t
\end{cases}
$$

After softmax, $e^{-\infty} = 0$, so every head — and therefore every output position — can only use the current and previous tokens.

### Scaling Uses the Head Size
The scale factor is $\sqrt{d_h}$, **not** $\sqrt{A}$. Each head's dot product sums over only $d_h$ dimensions, so its logits have variance $\approx d_h$; scaling by $\sqrt{A}$ would make the logits too small and the attention distribution too flat.

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input embeddings | $X$ | `(B, T, E)` | Shared input for all heads |
| Head projection weights | $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ | `(d_h, E)` | `nn.Linear(E, d_h, bias=False)` per head |
| Queries / Keys / Values per head | $Q_i, K_i, V_i$ | `(B, T, d_h)` | Each head's projected views |
| Scores per head | $Q_iK_i^\top / \sqrt{d_h}$ | `(B, T, T)` | Pairwise token similarities |
| Causal mask | $M$ | `(T, T)` | Broadcast over batch and heads |
| Attention weights per head | $\alpha_i$ | `(B, T, T)` | Rows sum to 1 over allowed keys |
| Head output | $\text{head}_i$ | `(B, T, d_h)` | $\alpha_i V_i$ |
| Concatenated heads | $C$ | `(B, T, A)` | `torch.cat(..., dim=2)` |
| Output projection | $W_O$ | `(A, A)` | `nn.Linear(A, A, bias=False)` |
| Final output | $O$ | `(B, T, A)` | Rounded to 4 decimals |

Where $B$ = batch size, $T$ = sequence length, and $A = H \cdot d_h$.

---

## 3. Vectorized Implementation & Numerical Stability

1. **`nn.ModuleList` for Parallel Heads:**
   - Each head is a full module with its own parameters, so they must be registered in a `ModuleList` (a plain Python list would not register parameters and they would never move with `.to(device)` or appear in `.parameters()`).
   - A list comprehension constructs all heads; each head then processes the **same** `embedded` tensor independently.

2. **Concatenation Axis:**
   - `torch.cat(head_outputs, dim=2)` joins along the feature dimension, producing `(B, T, H · d_h) = (B, T, A)`.
   - Concatenating on `dim=1` would splice sequences together along the time axis — a subtle but fatal shape bug.

3. **Output Projection Mixes Heads:**
   - `self.output_projection` is $A \times A$ and bias-free. It is the only operation that lets one head's features influence another head's output, and it also restores the original attention width.

4. **Determinism and the Seeding Quirk:**
   - Each `SingleHeadAttention.__init__` calls `torch.manual_seed(0)`, so **all heads with the same shape initialize with identical weights**. This is required by the problem template to reproduce the grader's expected output.
   - Consequence at initialization: every head computes the same attention pattern and the concatenated tensor is $H$ copies of one head's output. In a real model you would seed once *outside* the head construction (or use different seeds per head) so the heads can specialize. Here, preserving the template's seed placement is part of the contract — do not remove it.
   - The projections inside each head must be created in the order `key_gen`, `query_gen`, `value_gen`, since each `nn.Linear` consumes RNG state as it initializes.

5. **Numerical Stability Safeguards:**
   - `float("-inf")` for masked positions makes the post-softmax weight exactly `0`; a finite sentinel such as `-1e9` leaks a tiny nonzero weight.
   - Every row of the causal mask contains at least the diagonal, so no row is fully masked and softmax cannot produce `NaN`.
   - Softmax is applied on `dim=2` (the key axis) inside every head.
   - PyTorch's softmax subtracts the row maximum internally (log-sum-exp), keeping large logits stable.

6. **Shape Assumptions:**
   - `k.transpose(1, 2)` assumes exactly 3 dimensions `(B, T, d_h)`. It swaps time and head-feature axes. `transpose(-2, -1)` would be safer for general ranks but is not required here.
   - `attention_dim // num_heads` uses integer division. If `attention_dim` is **not divisible** by `num_heads`, the concatenated width $H \cdot \lfloor A/H \rfloor$ is smaller than $A$ and the output projection's input size will mismatch. The problem guarantees divisibility.

7. **Rounding Invariant:**
   - Only the final blended output is rounded, to **4 decimal places**: `torch.round(output, decimals=4)`.

---

## 4. Complexity Analysis

Let $B$ = batch size, $T$ = sequence length, $E$ = `embedding_dim`, $A$ = `attention_dim`, $H$ = `num_heads`, and $d_h = A/H$.

- **Time Complexity:** $O(BT^2A + BTEA + BTA^2)$
  - **Per-head projections:** $H$ heads each multiply `(B, T, E)` by `(E, d_h)`, totaling $O(BTEA)$.
  - **Per-head scores and weighted sums:** each head costs $O(BT^2d_h)$ for $QK^\top$ and the same for $\alpha V$; summed over $H$ heads this is $O(BT^2A)$.
  - **Head concatenation:** $O(BTA)$ copies.
  - **Output projection:** `(B, T, A) @ (A, A)` costs $O(BTA^2)$.
  - For long sequences ($T \gg A$) the quadratic attention term dominates.
- **Space Complexity:** $O(HBT^2 + BTA)$ during training, $O(BT^2 + BTA)$ peak during inference
  - Each head materializes a `(B, T, T)` attention matrix. If autograd retains all heads' activations for the backward pass, the combined attention memory is $H \cdot BT^2$. In eager inference each head's matrix can be freed after its output is produced, leaving one $BT^2$ matrix as the peak.
  - Head outputs total `(B, T, A)`, and the concatenated tensor plus final output add a few more `O(BTA)` buffers.
  - Parameters: $3EA$ for all Q/K/V projections plus $A^2$ for the output projection (no biases).

---

## 5. Edge Cases & Gotchas

- [x] **`attention_dim % num_heads != 0`:** integer division truncates `head_size`, the concatenated width falls short of `attention_dim`, and the output projection fails. Divisibility is a hard requirement.
- [x] **`num_heads = 1`:** degenerates to single-head attention followed by a square output projection; shapes remain valid.
- [x] **Single-token sequence (`T = 1`):** the mask is `[[1]]`; each head's softmax is `1.0`, so every head returns its $V$ and the layer reduces to the output projection of concatenated values.
- [x] **First token (`t = 0`):** all heads can only attend to position 0.
- [x] **Concatenation axis:** must be `dim=2` (features); `dim=1` silently mixes time steps.
- [x] **Device consistency:** the mask is created with `device=embedded.device` inside every head; a CPU mask on a GPU tensor raises a device error.
- [x] **Mask dtype:** `torch.ones` produces a float mask and `mask == 0` yields the boolean mask required by `masked_fill`; using `torch.ones(..., dtype=torch.bool)` and inverting with `~` is equivalent.
- [x] **Seeding side effect:** all heads start identical. This is intentional for reproducibility here, but it means multi-head diversity is zero at initialization; removing the seed changes the expected test output.
- [x] **Projection order:** `key_gen`, `query_gen`, `value_gen` must be declared in that order to match the seeded weight assignment.
- [x] **Transpose rank:** `transpose(1, 2)` only works for 3D tensors; it would be wrong if head inputs carried an extra dimension.
- [x] **Scaling factor:** divide by $\sqrt{d_h} = \sqrt{\texttt{k.shape[2]}}$, not by $\sqrt{A}$.
- [x] **No biases anywhere:** all four `nn.Linear` layers use `bias=False`.
- [x] **Rounding precision:** round after the output projection, never on per-head scores or attention weights.
- **Common pitfalls during implementation:**
  - Using a plain Python list instead of `nn.ModuleList`, which leaves the heads' parameters unregistered.
  - Forgetting `return scores @ v` in the single-head `forward`, leaving the head output undefined (the snippet in this note completes that line).
  - Normalizing softmax on the wrong axis.
  - Building the mask once outside the head but using the wrong sequence length for a different input.

---

## 6. Clean Code

```python
import torch
import torch.nn as nn
from torchtyping import TensorType


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        head_size = attention_dim // num_heads

        self.heads = nn.ModuleList([
            self.SingleHeadAttention(embedding_dim, head_size)
            for _ in range(num_heads)
        ])

        self.output_projection = nn.Linear(
            attention_dim, attention_dim, bias=False
        )

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        head_outputs = [head(embedded) for head in self.heads]

        # Concatenate along the feature axis: (B, T, H * head_size)
        concatenated = torch.cat(head_outputs, dim=2)

        output = self.output_projection(concatenated)

        return torch.round(output, decimals=4)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)

            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ k.transpose(1, 2)
            scores = scores / (k.shape[2] ** 0.5)

            context_length = k.shape[1]
            lower_triangular = torch.tril(
                torch.ones(
                    context_length,
                    context_length,
                    device=embedded.device
                )
            )

            scores = scores.masked_fill(
                lower_triangular == 0,
                float("-inf")
            )
            scores = nn.functional.softmax(scores, dim=2)

            return scores @ v
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Head Concatenation Example ($H = 2$, $A = 4$, $d_h = 2$, $T = 3$)

```text
X  (1, 3, E)  ── shared by both heads ──┐
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        ▼                                                               ▼
   Head 1 (d_h = 2)                                               Head 2 (d_h = 2)
   Q1, K1, V1  (1, 3, 2)                                          Q2, K2, V2  (1, 3, 2)
   scores1     (1, 3, 3)                                          scores2     (1, 3, 3)
   mask s <= t -> softmax                                          mask s <= t -> softmax
   out1 = α1 V1 (1, 3, 2)                                         out2 = α2 V2 (1, 3, 2)
        │                                                               │
        └───────────────────────────┬───────────────────────────────────┘
                                    ▼
                 cat(dim=2)  ->  C  (1, 3, 4)
                                    ▼
                 W_O: Linear(4, 4, bias=False)
                                    ▼
                 O  (1, 3, 4)  ->  round(., 4)
```

Per token, the concatenation looks like:

```text
token t:   [ out1_t[0], out1_t[1] | out2_t[0], out2_t[1] ]
              └── head 1 (2 dims) ┘ └── head 2 (2 dims) ──┘
```

Attention weights inside each head are still causal:

```text
     k0     k1     k2                k0     k1     k2
q0 [ 1.00,  0.00,  0.00 ]      q0 [ 1.00,  0.00,  0.00 ]
q1 [ 0.35,  0.65,  0.00 ]      q1 [ 0.55,  0.45,  0.00 ]   <- head 2 differs
q2 [ 0.15,  0.30,  0.55 ]      q2 [ 0.20,  0.20,  0.60 ]      only if seeds differ
```

### Architectural Pipeline

```text
                 X (B, T, E)
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
   head 1          head 2    ...    head H        (each: Q,K,V + causal attention)
   (B,T,d_h)       (B,T,d_h)       (B,T,d_h)
      └───────────────┼───────────────┘
                      ▼
        concat on feature axis -> (B, T, A)
                      ▼
        output projection W_O (A, A)
                      ▼
              O (B, T, A) -> round(., 4)
```

* **Next Review Date:**
* **Key Takeaway:** Multi-head attention splits the attention width into $H$ independent subspaces, runs causal scaled dot-product attention in each with scale $\sqrt{d_h}$, concatenates the outputs on the feature axis, and lets a final $A \times A$ projection mix the heads — with careful seeding because every head starts from the same template weights.

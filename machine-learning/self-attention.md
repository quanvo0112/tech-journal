# Self Attention

- **Problem Link:** [NeetCode - Self Attention](https://neetcode.io/problems/self-attention)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `Attention & Transformers`
- **Implementation Framework:** `PyTorch`
- **Core Concept / Formulation:** `Scaled Dot-Product Attention, Query/Key/Value Projections, Causal Masking, Softmax Normalization`
- **Last Practiced:** 2026-10-24
- **Proficiency Level:**
  - [ ] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Motivation
Self-attention lets every token build its representation by **looking at other tokens in the same sequence** and weighting them by relevance. Unlike a recurrent layer, all positions are processed in parallel: the only thing that encodes order and causality is the mask.

Given an embedded sequence $X \in \mathbb{R}^{T \times E}$, three learned linear projections produce queries, keys, and values:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

with $W_Q, W_K, W_V \in \mathbb{R}^{E \times A}$, where $E$ is `embedding_dim` and $A$ is `attention_dim`. All three projections are **bias-free**, so they are pure linear maps.

### Scaled Dot-Product Attention
$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{A}} + M\right) V
$$

- $QK^\top \in \mathbb{R}^{T \times T}$: entry $(t, s)$ is the dot product of query $t$ with key $s$ — a similarity score for "how much should token $t$ attend to token $s$".
- $1/\sqrt{A}$: **scaling factor**. A dot product of two $A$-dimensional vectors with roughly unit-variance entries has variance $\approx A$; without scaling, large logits push softmax into saturation, where gradients vanish. Dividing by $\sqrt{A}$ restores unit variance.
- $M$: the **causal mask**, $M_{ts} = 0$ when $s \le t$ and $M_{ts} = -\infty$ when $s > t$.
- Softmax is taken over the **key axis** (the last dimension), so each row of the attention matrix sums to $1$.

### Causal Masking
$$
M_{ts} =
\begin{cases}
0, & s \le t \quad \text{(past and current token allowed)} \\
-\infty, & s > t \quad \text{(future token blocked)}
\end{cases}
$$

After softmax, $e^{-\infty} = 0$, so future positions receive **exactly zero** weight. This is what makes the layer autoregressive: the output at position $t$ depends only on $X_0, \dots, X_t$, never on $X_{t+1}, \dots, X_{T-1}$.

### Output
$$
O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{A}} + M\right) V \in \mathbb{R}^{T \times A}
$$

Row $t$ is a convex combination of the value vectors of positions $0 \dots t$:

$$
O_t = \sum_{s=0}^{t} \alpha_{ts} V_s, \qquad \sum_{s=0}^{t} \alpha_{ts} = 1
$$

No positional encoding is applied inside this layer — the causal mask only restricts *which* tokens can be seen, not *where* they are.

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input embeddings | $X$ | `(B, T, E)` | Batch of embedded token sequences |
| Query weights | $W_Q$ | `(A, E)` | `nn.Linear(E, A, bias=False)` |
| Key weights | $W_K$ | `(A, E)` | `nn.Linear(E, A, bias=False)` |
| Value weights | $W_V$ | `(A, E)` | `nn.Linear(E, A, bias=False)` |
| Queries | $Q$ | `(B, T, A)` | Per-token query vectors |
| Keys | $K$ | `(B, T, A)` | Per-token key vectors |
| Values | $V$ | `(B, T, A)` | Per-token value vectors |
| Raw scores | $S$ | `(B, T, T)` | $QK^\top / \sqrt{A}$ before masking |
| Causal mask | $M$ | `(T, T)` | Lower-triangular; broadcasts over the batch |
| Attention weights | $\alpha$ | `(B, T, T)` | Softmax over the key axis; rows sum to 1 |
| Output | $O$ | `(B, T, A)` | Contextualized token representations |

Where $B$ = batch size, $T$ = sequence length, $E$ = `embedding_dim`, $A$ = `attention_dim`. Note that the output's last dimension is $A$, which may differ from $E$.

---

## 3. Vectorized Implementation & Numerical Stability

1. **Batched Matrix Multiplication:**
   - `Q @ K.transpose(-2, -1)` uses PyTorch's batched `matmul`, computing all $B$ score matrices in one call. `transpose(-2, -1)` swaps the last two axes for every batch element, giving $K^\top \in \mathbb{R}^{B \times A \times T}$, so the product is `(B, T, A) @ (B, A, T) = (B, T, T)`.
   - No Python loops over batch, heads, or positions.

2. **Mask Broadcasting:**
   - The mask is built as `(T, T)` and implicitly broadcast to `(B, T, T)`. Every sequence in the batch shares the same causal structure, so allocating one mask is sufficient.
   - `torch.tril(torch.ones(T, T))` keeps the diagonal and everything below it. `masked_fill(mask == 0, -inf)` blocks the strict upper triangle.

3. **Softmax Axis Is Critical:**
   - `torch.softmax(scores, dim=2)` normalizes across the **key** dimension. Normalizing on `dim=1` would instead normalize across queries and produce a completely different, wrong operation.
   - Every masked row still has at least one finite entry — row $t$ always contains the diagonal $s = t$ — so no row is entirely $-\infty$. This guarantees softmax never produces `NaN`.

4. **Numerical Stability Safeguards:**
   - Use a true $-\infty$ (`float("-inf")`) rather than a large negative constant. A finite value such as `-1e9` leaves a tiny nonzero weight after softmax; $-\infty$ maps to exactly `0`.
   - The $\sqrt{A}$ scaling keeps the logits in a numerically healthy range, avoiding `exp` overflow and vanishing gradients.
   - PyTorch's `softmax` internally subtracts the row maximum (log-sum-exp trick), so even moderately large logits remain stable.

5. **Determinism and Initialization Order:**
   - `torch.manual_seed(0)` in `__init__` makes the randomly initialized weights reproducible, which the grader relies on to compare outputs.
   - The projections **must be created in the order `key`, `query`, `value`**, because each `nn.Linear` consumes the RNG state when its parameters are initialized. Reordering the declarations changes the weights and therefore the expected output.
   - All three layers use `bias=False`; adding biases would shift every projection and break parity with the reference.

6. **Rounding Invariant:**
   - The final output is rounded to **4 decimal places** via `torch.round(output, decimals=4)`; rounding must happen after the weighted sum, never on intermediate scores.

---

## 4. Complexity Analysis

Let $B$ = batch size, $T$ = sequence length, $E$ = `embedding_dim`, $A$ = `attention_dim`.

- **Time Complexity:** $O(BT^2A + BTAE)$
  - Projections: three matmuls of `(B, T, E)` by `(E, A)`, each $O(BTEA)$.
  - Scores: `Q @ K^T` is $O(BT^2A)$ — for every one of the $T^2$ query/key pairs, a length-$A$ dot product.
  - Masked softmax: $O(BT^2)$.
  - Weighted sum: `(B, T, T) @ (B, T, A)` is another $O(BT^2A)$, which usually dominates because $T \gg E, A$ in long sequences.
- **Space Complexity:** $O(BT^2 + BTA)$
  - The attention score/weight matrix is $O(BT^2)$ — the quadratic term that motivates efficient attention variants for long contexts.
  - $Q$, $K$, $V$, and the output are $O(BTA)$ each; activations needed for backpropagation are retained when gradients are enabled.

---

## 5. Edge Cases & Gotchas

- [x] **Single-token sequence (`T = 1`):** the mask is `[[1]]`, softmax over one element yields `1.0`, and the output is exactly $V_0$. Causal attention degenerates to a copy of the value projection.
- [x] **First token (`t = 0`):** only the diagonal is unmasked, so the first output is always $V_0$ regardless of the rest of the sequence.
- [x] **Future leakage:** must use `tril` (inclusive diagonal). Excluding the diagonal would prevent a token from attending to itself; using no mask or a full mask would leak future tokens.
- [x] **Fully masked row:** impossible here because the diagonal is always allowed. A row of all $-\infty$ would make softmax divide by zero and return `NaN`.
- [x] **Device consistency:** the mask is allocated with `device=embedded.device`; a CPU mask on a CUDA tensor would raise a device mismatch during `masked_fill`.
- [x] **Batch broadcasting:** the `(T, T)` mask must broadcast against `(B, T, T)`; keeping it 2D is correct and avoids an unnecessary $B\times$ allocation.
- [x] **`attention_dim != embedding_dim`:** the output is `(B, T, A)`, not `(B, T, E)`. Downstream residual connections require $A = E$, but this layer itself does not assume it.
- [x] **Seed scope:** the seed is set per instance, so constructing a second `SingleHeadAttention` re-seeds to `0` and reproduces the same initial weights. Initialization order is part of the contract.
- [x] **Rounding precision:** round only the final output to 4 decimals; rounding logits or attention weights would change the result.
- [x] **No bias:** `bias=False` is required; a bias term would add a constant to every projection and shift the scores.
- **Common pitfalls during implementation:**
  - Forgetting to transpose $K$, which makes the matmul shapes incompatible.
  - Scaling by $\sqrt{A}$ after the mask (harmless numerically but conceptually wrong) or using `A` instead of $\sqrt{A}$.
  - Using `dim=1` in `softmax`.
  - Using a finite negative sentinel instead of `-inf`.
  - Reordering the `key` / `query` / `value` declarations, which silently changes the seeded initialization.

---

## 6. Clean Code

```python
import math

import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        # Scaled dot-product scores: (B, T, A) @ (B, A, T) -> (B, T, T)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        seq_len = embedded.size(1)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=embedded.device)
        )

        # Block future positions before softmax.
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Normalize over the key axis.
        attention_weights = torch.softmax(scores, dim=2)

        # Weighted sum of values.
        output = attention_weights @ V

        return torch.round(output, decimals=4)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Masked Attention Example ($T = 3$)

Raw scaled scores and the causal mask:

```text
scores (B=1)              mask (T=3)
       k0    k1    k2            k0  k1  k2
q0 [  2.0,  1.0,  3.0 ]      q0 [ 1,  0,  0 ]
q1 [  1.0,  2.0,  0.0 ]      q1 [ 1,  1,  0 ]
q2 [  0.5,  1.5,  2.5 ]      q2 [ 1,  1,  1 ]
```

After `masked_fill(mask == 0, -inf)`:

```text
       k0     k1     k2
q0 [  2.0,  -inf,  -inf ]
q1 [  1.0,   2.0,  -inf ]
q2 [  0.5,   1.5,   2.5 ]
```

Softmax over keys (rows sum to 1, future weights are exactly 0):

```text
       k0      k1      k2
q0 [ 1.000,  0.000,  0.000 ]   -> O0 = V0
q1 [ 0.269,  0.731,  0.000 ]   -> O1 = 0.269 V0 + 0.731 V1
q2 [ 0.090,  0.245,  0.665 ]   -> O2 = 0.090 V0 + 0.245 V1 + 0.665 V2
```

Output shape is `(B, T, A)`. Each row is a convex combination of the values of **only the current and previous tokens**.

---

### Architectural Pipeline

```text
        X (B, T, E)
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
   W_Q     W_K      W_V
     ▼       ▼        ▼
   Q (B,T,A) K (B,T,A) V (B,T,A)
     │       │
     └── Q @ K^T ──> scores (B, T, T)
                       │
              ÷ sqrt(A) (scale)
                       │
        + causal mask (T,T): keep s <= t
                       │
              softmax(dim = keys)
                       │
              weights (B, T, T)
                       │
                 weights @ V
                       │
                 O (B, T, A)  -> round(., 4)
```

* **Next Review Date:**
* **Key Takeaway:** Self-attention is just three bias-free projections followed by `softmax(QK^T / sqrt(A)) V`; the lower-triangular mask applied *before* softmax is what enforces causality, and using `-inf` guarantees future positions get exactly zero weight.

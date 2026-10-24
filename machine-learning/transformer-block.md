# Transformer Block

- **Problem Link:** [NeetCode - Transformer Block](https://neetcode.io/problems/transformer-block)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `Attention & Transformers`
- **Implementation Framework:** `PyTorch`
- **Core Concept / Formulation:** `Pre-LayerNorm Transformer Block, Multi-Head Self-Attention, Residual Connections, Position-Wise Feed-Forward Network, Dropout`
- **Last Practiced:** 2026-10-24
- **Proficiency Level:**
  - [ ] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Motivation
A Transformer block stacks two residual sub-layers: **multi-head self-attention** (mixes information across positions) and a **position-wise feed-forward network** (transforms each position independently). Residual connections preserve an identity path for gradients, while normalization keeps activations well-scaled.

This implementation uses the **Pre-LN** layout: each sub-layer receives a *normalized* copy of the stream, and its output is added back to the **un-normalized** residual stream.

$$
x_1 = x + \operatorname{MHA}\!\left(\operatorname{LN}_1(x)\right)
$$

$$
x_2 = x_1 + \operatorname{FFN}\!\left(\operatorname{LN}_2(x_1)\right)
$$

The block output is $x_2$, rounded to 4 decimals.

### Layer Normalization
For each token vector $h \in \mathbb{R}^{D}$ (normalization is over the **feature** dimension, independently per position and per batch element):

$$
\mu = \frac{1}{D}\sum_{j=1}^{D} h_j, \qquad
\sigma^2 = \frac{1}{D}\sum_{j=1}^{D}(h_j - \mu)^2
$$

$$
\operatorname{LN}(h) = \gamma \odot \frac{h - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

with learnable $\gamma, \beta \in \mathbb{R}^{D}$ initialized to $1$ and $0$, and $\epsilon = 10^{-5}$ by default. Because the statistics are computed per token, LayerNorm is independent of batch size and sequence length — the property that makes it suitable for autoregressive Transformers.

### Multi-Head Self-Attention Sub-Layer
With $H$ heads of size $d_h = D/H$:

$$
\text{head}_i = \operatorname{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_h}} + M\right)V_i
$$

$$
\operatorname{MHA}(z) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_H)\,W_O
$$

where $z = \operatorname{LN}_1(x)$, each head projects from $D$ to $d_h$, and $M$ is the causal mask ($M_{ts} = 0$ for $s \le t$, $-\infty$ otherwise). All attention projections and $W_O$ are **bias-free**.

### Position-Wise Feed-Forward Network
$$
\operatorname{FFN}(u) = \operatorname{Dropout}\!\left(W_2\,\operatorname{ReLU}(W_1u + b_1) + b_2\right)
$$

with $W_1 \in \mathbb{R}^{4D \times D}$, $W_2 \in \mathbb{R}^{D \times 4D}$, and dropout probability $p = 0.2$. The hidden width is $4D$, the standard expansion ratio from the original Transformer. The network is *position-wise*: the same $W_1, W_2$ are applied to every token, so there is no mixing across the time axis here.

Dropout during training zeroes each activation with probability $p$ and rescales survivors by $1/(1-p)$ so the expected value is preserved; at evaluation time it is the identity.

### Pre-LN vs. Post-LN
- **Pre-LN (used here):** $x + \text{Sublayer}(\operatorname{LN}(x))$. The residual path is a clean identity, so gradients flow directly to earlier layers. Training is more stable and typically does not require a learning-rate warmup.
- **Post-LN (original paper):** $\operatorname{LN}(x + \text{Sublayer}(x))$. Normalization sits on the residual path, which can make deep stacks harder to optimize without warmup.

The residual additions require the sub-layer output to match the stream width $D$.

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input stream | $x$ | `(B, T, D)` | Token representations entering the block |
| Normalized input | $\operatorname{LN}_1(x)$ | `(B, T, D)` | Per-token normalized copy |
| Per-head Q/K/V | $Q_i, K_i, V_i$ | `(B, T, d_h)` | $d_h = D/H$ |
| Per-head scores | $Q_iK_i^\top/\sqrt{d_h}$ | `(B, T, T)` | Causal pairwise similarities |
| Causal mask | $M$ | `(T, T)` | Broadcast over batch and heads |
| Per-head output | $\text{head}_i$ | `(B, T, d_h)` | $\alpha_iV_i$ |
| Concatenated heads | $C$ | `(B, T, D)` | `cat(dim=2)` restores width $D$ |
| Attention output | $\operatorname{MHA}$ | `(B, T, D)` | After bias-free output projection |
| First residual | $x_1$ | `(B, T, D)` | $x + \operatorname{MHA}(\operatorname{LN}_1(x))$ |
| Normalized stream | $\operatorname{LN}_2(x_1)$ | `(B, T, D)` | Input to the FFN |
| FFN hidden | $W_1u + b_1$ | `(B, T, 4D)` | Expanded representation |
| FFN output | $\operatorname{FFN}$ | `(B, T, D)` | After down-projection and dropout |
| Block output | $x_2$ | `(B, T, D)` | $x_1 + \operatorname{FFN}(\operatorname{LN}_2(x_1))$ |

Where $B$ = batch size, $T$ = sequence length, $D$ = `model_dim`, $H$ = `num_heads`, and $d_h = D/H$.

---

## 3. Vectorized Implementation & Numerical Stability

1. **Pre-LN Ordering:**
   - `layer_norm_1(embedded)` is passed into attention; the residual adds the **original** `embedded`, not the normalized tensor.
   - The second sub-layer normalizes `x` (the first residual result), passes it through the FFN, then adds the FFN output back to that same `x`.
   - Swapping to Post-LN (normalizing after the residual add) changes every value and the expected test output.

2. **LayerNorm Is Per-Token:**
   - `nn.LayerNorm(D)` normalizes the last dimension for every `(b, t)` pair independently. No statistics are shared across the sequence or batch, so varying sequence lengths and batch sizes do not affect the result.
   - Affine parameters $\gamma, \beta$ are learnable and initialized deterministically to ones/zeros, so their construction does not consume RNG state.

3. **Attention Internals:**
   - `q @ k.transpose(1, 2)` assumes a 3D `(B, T, d_h)` tensor.
   - Scaling by $\sqrt{d_h} = \sqrt{\texttt{k.shape[2]}}$ matches the per-head dot-product dimension; using $\sqrt{D}$ would flatten the distribution too much.
   - The causal mask is applied **before** softmax with `-inf`, so future positions get exactly zero weight.
   - Keys and queries are projected from the same normalized input; values too.

4. **FFN Expansion and Dropout:**
   - The 4× expansion ($D \to 4D \to D$) gives the block most of its parameter count and nonlinear capacity.
   - `nn.ReLU()` is stateless, so one shared instance is reused.
   - `nn.Dropout(0.2)` is applied **inside** the FFN, after the down-projection. The residual add happens after dropout, outside the FFN module.
   - **Dropout mode matters:** `nn.Module` is in training mode by default, so calling the block without `.eval()` applies dropout and rescales survivors. In evaluation mode, dropout is a no-op and the output changes. The expected test output depends on the mode used by the grader.

5. **Determinism and Initialization Order:**
   - `torch.manual_seed(0)` appears in every `__init__` *and* in `VanillaNeuralNetwork.forward`.
   - Each `SingleHeadAttention` reseeds to $0$, so all heads again initialize with identical weights; this is required for reproducibility in this exercise.
   - The FFN's forward reseeds to $0$ before the projections (which do not consume RNG) so the **dropout mask is deterministic** on every forward call. Remove it and repeated calls produce different outputs while in training mode.
   - Construction order is **Attention → Feed-Forward → LayerNorm 1 → LayerNorm 2**. Attention and FFN construction consume RNG (the heads reseed repeatedly, and the FFN's linear layers draw from the seeded generator), so changing the order or the internal declaration order of the layers changes the weights. The LayerNorms are deterministic (ones/zeros) but are kept in this order to match the template.
   - Keep `bias=False` on all four attention linears; the FFN linears keep their default biases, and LayerNorm keeps its affine parameters.

6. **Numerical Stability Safeguards:**
   - LayerNorm's $\epsilon$ prevents division by zero when a token's features are all equal.
   - `-inf` masking plus a diagonal that is always allowed guarantees no fully masked softmax row and no `NaN`.
   - PyTorch softmax uses the log-sum-exp trick internally, keeping large logits stable.
   - Residual connections prevent the repeated sub-layers from collapsing the activation scale.

7. **Rounding Invariant:**
   - Only the final block output is rounded, to **4 decimal places**: `torch.round(x, decimals=4)`.

---

## 4. Complexity Analysis

Let $B$ = batch size, $T$ = sequence length, $D$ = `model_dim`, $H$ = `num_heads`, $d_h = D/H$.

- **Time Complexity:** $O(BT^2D + BTD^2)$
  - **LayerNorm 1 / 2:** $O(BTD)$ each — two passes over the features per token.
  - **Attention projections:** $3 \cdot O(BTD \cdot d_h)$ per head, totaling $O(BTD^2)$ across heads; the output projection adds another $O(BTD^2)$.
  - **Attention scores and weighted sums:** $H$ heads each cost $O(BT^2d_h)$, totaling $O(BT^2D)$.
  - **FFN:** up-projection $O(BTD \cdot 4D)$ and down-projection $O(BT \cdot 4D \cdot D)$, totaling $O(BTD^2)$ with a constant factor of $8$.
  - **Residuals / dropout:** $O(BTD)$.
  - For long sequences the quadratic $BT^2D$ attention term dominates; for short sequences the $BTD^2$ projections dominate.
- **Space Complexity:** $O(HBT^2 + BTD)$ activations
  - Each head materializes a `(B, T, T)` score/attention matrix. With autograd retaining all heads for the backward pass, this is $H \cdot BT^2$; during inference each head's matrix can be released after use.
  - The FFN hidden tensor is `(B, T, 4D)`, the largest linear buffer.
  - Residuals and normalized copies are a few more `(B, T, D)` buffers.
  - Parameters: $4D^2$ for attention (3 QKV + output projection), $8D^2 + 5D$ for the FFN (weights + biases), and $4D$ for the two LayerNorms.

---

## 5. Edge Cases & Gotchas

- [x] **`model_dim % num_heads != 0`:** integer division truncates `d_h`; the concatenated width no longer matches $D$ and the output projection fails. Divisibility is required.
- [x] **Single-token sequence (`T = 1`):** attention collapses to each head's value vector, and the FFN still applies per position; shapes stay `(B, 1, D)`.
- [x] **Dropout in training vs. eval:** the block defaults to training mode, so dropout is active and survivors are scaled by $1/(1-0.2) = 1.25$. Calling `.eval()` disables dropout and yields different values. The grader's expected output assumes a specific mode.
- [x] **Seed inside `forward`:** `torch.manual_seed(0)` in the FFN's forward makes the dropout mask reproducible, but it also resets global RNG state — a side effect that would interfere with other stochastic layers if the block were embedded in a larger model.
- [x] **Identical heads at initialization:** each head reseeds to $0$, so attention heads start with the same weights. This is an artifact of the deterministic test contract; in production you would seed once outside the heads.
- [x] **Residual shape mismatch:** both residual adds require the sub-layer output to be `(B, T, D)`. If attention used a different total width or the FFN down-projected to a different size, the add would broadcast or fail.
- [x] **Wrong normalization placement:** applying LayerNorm *after* the residual (Post-LN) changes all values and loses the clean residual highway.
- [x] **Masked future positions:** the causal mask blocks future tokens inside attention only; the FFN is position-wise and needs no mask.
- [x] **No device argument on the mask:** `torch.ones(context_length, context_length)` creates a CPU tensor. On CUDA inputs, pass `device=embedded.device` (or `device=k.device`) to avoid a device mismatch during `masked_fill`.
- [x] **Attention bias flags:** all attention projections and the output projection are bias-free, while the FFN linears keep biases and LayerNorm keeps affine parameters. Mixing these conventions changes the parameter count and the outputs.
- [x] **Rounding precision:** round only the final block output, never the intermediate normalized or attention tensors.
- **Common pitfalls during implementation:**
  - Adding the normalized tensor to the residual instead of the original stream.
  - Forgetting the second residual connection.
  - Reversing the FFN order (down-projection before up-projection) or omitting ReLU.
  - Reordering the layer construction, which changes the seeded initialization.
  - Using one LayerNorm for both sub-layers instead of two independent ones.

---

## 6. Clean Code

```python
import torch
import torch.nn as nn
from torchtyping import TensorType


class TransformerBlock(nn.Module):
    class SingleHeadAttention(nn.Module):
        def __init__(self, model_dim: int, head_size: int):
            super().__init__()
            torch.manual_seed(0)

            self.key_gen = nn.Linear(model_dim, head_size, bias=False)
            self.query_gen = nn.Linear(model_dim, head_size, bias=False)
            self.value_gen = nn.Linear(model_dim, head_size, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ k.transpose(1, 2)
            scores = scores / (k.shape[2] ** 0.5)

            context_length = k.shape[1]
            lower_triangular = torch.tril(
                torch.ones(context_length, context_length)
            )

            scores = scores.masked_fill(
                lower_triangular == 0, float("-inf")
            )
            scores = nn.functional.softmax(scores, dim=2)

            return scores @ v

    class MultiHeadedSelfAttention(nn.Module):
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)

            self.att_heads = nn.ModuleList()

            for _ in range(num_heads):
                self.att_heads.append(
                    self.SingleHeadAttention(
                        model_dim, model_dim // num_heads
                    )
                )

            self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            head_outputs = [
                head(embedded) for head in self.att_heads
            ]

            concatenated = torch.cat(head_outputs, dim=2)

            return self.output_proj(concatenated)

    class VanillaNeuralNetwork(nn.Module):
        def __init__(self, model_dim: int):
            super().__init__()
            torch.manual_seed(0)

            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)

            x = self.up_projection(x)
            x = self.relu(x)
            x = self.down_projection(x)

            return self.dropout(x)

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
        self.feed_forward = self.VanillaNeuralNetwork(model_dim)
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Pre-LN attention sub-layer with residual.
        attention_out = self.attention(self.layer_norm_1(embedded))
        x = embedded + attention_out

        # Pre-LN feed-forward sub-layer with residual.
        ff_out = self.feed_forward(self.layer_norm_2(x))
        x = x + ff_out

        return torch.round(x, decimals=4)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Pre-LN Block Dataflow ($D = 4$, $H = 2$, $T = 3$)

```text
                 x (1, 3, 4)  ──────────────────────────────┐  residual
                        │                                   │
                        ▼                                   │
                  LayerNorm_1                               │
                        │ (1, 3, 4)                         │
                        ▼                                   │
              Multi-Head Self-Attention                     │
        ┌───────────────┴────────────────┐                  │
        │ head 1 (d_h=2)   head 2 (d_h=2)│                  │
        │   causal attn     causal attn  │                  │
        └───────────────┬────────────────┘                  │
                        ▼                                   │
                cat(dim=2) (1, 3, 4)                        │
                        ▼                                   │
              output_proj (4 -> 4)                          │
                        │ (1, 3, 4)                         │
                        ▼                                   ▼
                        └──────────────► (+) ──────────> x1 (1, 3, 4)
                                            ▲
                                            │ residual
                        ┌───────────────────┘
                        ▼
                  LayerNorm_2
                        │ (1, 3, 4)
                        ▼
              up_projection (4 -> 16)
                        │ (1, 3, 16)
                        ▼
                      ReLU
                        ▼
             down_projection (16 -> 4)
                        │ (1, 3, 4)
                        ▼
                   Dropout(0.2)
                        │ (1, 3, 4)
                        ▼
                        └──────────────► (+) ──────────> x2 (1, 3, 4)
                                                                  │
                                                            round(., 4)
                                                                  ▼
                                                            output (1, 3, 4)
```

The two residual additions both operate on the **unnormalized stream**, which is what makes Pre-LN train stably: gradients can travel from the loss back to the input through pure addition, without passing through a normalization or a nonlinearity.

### Architectural Pipeline

```text
      Input x
         │
    ┌────┴──────────────────────────────────────┐
    │  x = x + MHA(LayerNorm_1(x))              │  attention sub-layer
    └────┬──────────────────────────────────────┘
         │
    ┌────┴──────────────────────────────────────┐
    │  x = x + FFN(LayerNorm_2(x))              │  feed-forward sub-layer
    │      FFN: D -> 4D -> ReLU -> D -> Dropout │
    └────┬──────────────────────────────────────┘
         │
    round(x, 4)
         │
      Output
```

* **Next Review Date:**
* **Key Takeaway:** A Pre-LN Transformer block is two residual sub-layers — `x + MHA(LN(x))` then `x + FFN(LN(x))` — where attention mixes information across positions and the 4× feed-forward network transforms each position independently; the residual stream stays unnormalized, and the exact seeded initialization plus dropout mode determine the expected output.

# Sentiment Analysis

- **Problem Link:** [NeetCode - Sentiment Analysis](https://neetcode.io/problems/sentiment-analysis)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `NLP`
- **Implementation Framework:** `PyTorch`
- **Core Concept / Formulation:** `Neural Bag of Words (Mean-Pooled Embeddings + Linear + Sigmoid Binary Classifier)`
- **Last Practiced:**
- **Proficiency Level:**
  - [ ] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Objective
Sentiment analysis is binary text classification: given a padded batch of token-ID sequences, predict the probability that each sentence is positive. The model here is the classic **Neural Bag of Words (NBOW)** — the simplest neural text classifier and a direct extension of the `Word Embeddings` note.

The pipeline is: token IDs $\to$ dense embeddings $\to$ sentence vector (mean pool) $\to$ linear projection $\to$ sigmoid probability.

### Key Equations
Let $x \in \mathbb{Z}^{B \times T}$ be a batch of $B$ sentences, each padded/truncated to $T$ tokens, and let $E \in \mathbb{R}^{V \times d}$ be the embedding matrix with $d = 16$.

**1. Embedding lookup (gather over the token dimension):**
$$Z_{b,t,:} = E_{x_{b,t},:}, \qquad Z \in \mathbb{R}^{B \times T \times d}$$

**2. Mean pooling over tokens (produces one vector per sentence):**
$$h_b = \frac{1}{T} \sum_{t=1}^{T} Z_{b,t,:}, \qquad h \in \mathbb{R}^{B \times d}$$

**3. Linear projection to a scalar logit:**
$$\text{logit}_b = h_b W^\top + c, \qquad W \in \mathbb{R}^{1 \times d},\; c \in \mathbb{R}$$

**4. Sigmoid $\to$ probability:**
$$p_b = \sigma(\text{logit}_b) = \frac{1}{1 + e^{-\text{logit}_b}}, \qquad p_b \in (0, 1)$$

**5. Rounding (problem I/O contract):**
$$\hat{p}_b = \operatorname{round}(p_b,\, 4)$$

### Why Mean Pooling and Sigmoid
- **Mean pooling** collapses the variable-length token axis into a fixed $d$-dimensional "sentence fingerprint", making the classifier permutation-invariant (word order is discarded — the "bag of words" assumption).
- **Sigmoid** squashes the unbounded logit into $(0, 1)$, giving a valid probability for binary classification. A single output unit + sigmoid is the standard encoding of a Bernoulli log-likelihood target.

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input token IDs | $x$ | `(B, T)` | Padded batch of integer token IDs |
| Embedding weight | $E$ | `(V, 16)` | Learnable lookup table (`nn.Embedding`) |
| Embedded batch | $Z$ | `(B, T, 16)` | Dense vector per token |
| Pooled sentence vector | $h$ | `(B, 16)` | Mean over token dimension (`dim=1`) |
| Linear weight | $W$ | `(1, 16)` | Projects 16 $\to$ 1 (`nn.Linear(16, 1)`) |
| Logits | $\text{logit}$ | `(B, 1)` | Pre-activation scores |
| Predictions | $p$ | `(B, 1)` | Sigmoid probabilities, rounded to 4 decimals |

Where $B$ = batch size, $T$ = padded sequence length, $V$ = vocabulary size, $d = 16$ = embedding dimension.

---

## 3. Vectorized Implementation & Numerical Stability

1. **Why fully vectorized (no Python loops):**
   - `nn.Embedding` performs the entire `(B, T)` gather in one C++/CUDA kernel — no per-token indexing loop.
   - `embeddings.mean(dim=1)` reduces the token axis in a single fused pass, replacing the naive `for t in range(T): acc += ...`.
   - `nn.Linear` is one GEMM (`(B,16) @ (16,1)^T`), running on optimized BLAS.
   - The whole forward pass is three tensor ops — this scales to large batches with no interpreter overhead.

2. **Shape discipline (the main source of bugs):**
   - Pooling must use `dim=1` (the token axis), **not** `dim=0` (batch) or `dim=-1` (features). Using `dim=-1` would mix feature channels and collapse the 16-dim vector to a scalar per token — a subtle but fatal error.
   - The result of `mean(dim=1)` is `(B, 16)`, exactly matching `nn.Linear(16, 1)`'s expected `in_features`.

3. **Numerical stability:**
   - Sigmoid saturates gracefully: for large $|z|$ it approaches $0$/$1$ without overflow (unlike naive `1/(1+exp(-z))` only in extreme edge cases; PyTorch's implementation is stable).
   - `torch.round(..., decimals=4)` only affects the returned values, not the training gradients — it is a pure output-formatting step required by the grader.

4. **Reproducibility:**
   - `torch.manual_seed(0)` fixes embedding/linear initialization so the rounded outputs are deterministic across runs (necessary for exact-match grading).

---

## 4. Complexity Analysis

Let $B$ = batch size, $T$ = sequence length, $d = 16$ = embedding dim.

- **Time Complexity:** $O(B \cdot T \cdot d)$
  - Dominated by the embedding gather plus the mean reduction, both touching all $B \cdot T \cdot d$ elements once. The final linear layer is a negligible $O(B \cdot d)$.
- **Space Complexity / Activation Memory:** $O(B \cdot T \cdot d)$
  - The embedded tensor $Z$ `(B, T, 16)` is the largest allocation; the pooled `(B, 16)` and logits `(B, 1)` are trivial by comparison. All of $Z$ is cached for the backward pass when training.

---

## 5. Edge Cases & Gotchas

- [ ] **Padding pollutes the mean:** `mean(dim=1)` averages over **all** $T$ positions, including padded `0`-ID embeddings. Since padding is ID `0`, its (learned) embedding vector is included in the average — acceptable for this fixed-length setup, but the single most important caveat when lengths vary.
- [ ] **Wrong reduction axis:** `dim=1` (tokens) is required; `dim=0` or `dim=-1` silently breaks the model.
- [ ] **Output shape must be `(B, 1)`:** keep the trailing singleton dim — do not `.squeeze()` it away if the grader expects `(B, 1)`.
- [ ] **Rounding to 4 decimals:** `decimals=4` exactly as specified; more or fewer decimals fail the match.
- [ ] **Determinism:** keep `torch.manual_seed(0)` so initialization (and thus the rounded predictions) is reproducible.
- [ ] Common pitfalls during implementation:
  - Using `nn.Embedding`'s `padding_idx` is *not* set here — every ID including `0` has a trainable vector (intentional per this solution).
  - Forgetting `super().__init__()` inside the custom `nn.Module`, which breaks parameter registration.

---

## 6. Clean Code

```python
import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        embeddings = self.embedding(x)       # (B, T, 16)
        pooled = embeddings.mean(dim=1)      # (B, 16)
        logits = self.linear(pooled)         # (B, 1)
        predictions = self.sigmoid(logits)   # (B, 1)
        return torch.round(predictions, decimals=4)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Visual Forward Pass

```text
Input token IDs x            (B, T)
        ↓  nn.Embedding(V, 16)
Embedded tokens Z            (B, T, 16)
        ↓  .mean(dim=1)   <-- collapse the token axis
Sentence vectors h           (B, 16)
        ↓  nn.Linear(16, 1)
Logits                       (B, 1)
        ↓  nn.Sigmoid
Probabilities p in (0, 1)    (B, 1)
        ↓  torch.round(..., 4)
Final predictions            (B, 1)
```

### Worked Mini-Example

```text
B = 2, T = 3, d = 16

Z = [
      [ e(good), e(film), e(<pad>) ],   -> mean over t -> h_0 (16,)
      [ e(bad),  e(plot), e(boring) ]   -> mean over t -> h_1 (16,)
    ]                                   (B, T, 16) -> (B, 16)

logits = h @ W^T + c        (B, 16) @ (16, 1) -> (B, 1)
p      = sigmoid(logits)    -> e.g. [[0.8123], [0.1047]]
round(p, 4)                 -> [[0.8123], [0.1047]]   (final output)
```

* **Next Review Date:**
* **Key Takeaway:** The neural bag-of-words classifier is just `Embedding -> mean over tokens -> Linear -> Sigmoid` — remember to pool along `dim=1`, keep the `(B, 1)` output, and note that the mean silently includes padded `0`-ID positions.

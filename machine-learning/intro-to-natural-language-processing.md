# Intro to Natural Language Processing

- **Problem Link:** [NeetCode - Intro to Natural Language Processing](https://neetcode.io/problems/nlp-intro)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `NLP`
- **Implementation Framework:** `PyTorch`
- **Core Concept / Formulation:** `Text Tokenization, Vocabulary Construction, Integer Encoding, Sequence Padding`
- **Last Practiced:**
- **Proficiency Level:**
  - [ ] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Objective
Neural networks cannot consume raw strings — every text pipeline must first convert sentences into rectangular tensors of integers. `get_dataset` performs the three canonical preprocessing steps shared by virtually all NLP models (bag-of-words, RNNs, transformers):

1. **Vocabulary Construction:** collect every unique token across all sentences and assign each an integer ID.
2. **Integer Encoding:** replace each token in a sentence by its ID, preserving word order.
3. **Sequence Padding:** pad every sentence to the length of the longest one so all rows share a common dimension.

Given the combined corpus $C = P \,\|\, N$ (positive sentences followed by negative sentences), define the vocabulary as the sorted set of tokens:

$$V = \operatorname{sorted}\bigl(\{\, w \mid w \in \operatorname{split}(s),\; s \in C \,\}\bigr)$$

### Key Equations
The word-to-index map reserves `0` for padding, so IDs begin at `1`:

$$\operatorname{w2i}(w) = 1 + \operatorname{index}_{\,V}(w), \qquad \operatorname{w2i} : V \to \{1, \dots, |V|\}$$

Encoding maps a tokenized sentence of length $\ell_s$ to an integer sequence:

$$\operatorname{enc}(s) = \bigl[\, \operatorname{w2i}(w_1),\, \operatorname{w2i}(w_2),\, \dots,\, \operatorname{w2i}(w_{\ell_s}) \,\bigr] \in \mathbb{Z}^{\ell_s}$$

Padding builds the final matrix $X \in \mathbb{R}^{N \times L_{\max}}$ where $L_{\max} = \max_s \ell_s$:

$$X_{i,j} = \begin{cases} \operatorname{enc}(s_i)_j & j < \ell_{s_i} \\ 0 & \text{otherwise (padding)} \end{cases}$$

### Why `float32` and Why `0` Padding
- The problem declares `TensorType[float]`, so token IDs are stored as `float32` even though they are conceptually integers — downstream embedding layers accept either, but the return contract must be honored.
- `pad_sequence` defaults to `padding_value = 0`, which is exactly why the vocabulary deliberately leaves ID `0` free. This reserve guarantees padding can never collide with a real token.

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Combined corpus | $C$ | `list[str]` of length $N$ | `positive` sentences followed by `negative` sentences |
| Vocabulary | $V$ | `list[str]` of length $\lvert V\rvert$ | Sorted unique tokens; IDs assigned as $1 \dots \lvert V\rvert$ |
| Word-to-index | `word_to_id` | `dict[str, int]` | Maps token $\to$ integer ID (1-based) |
| Per-sentence tensor | $t_i$ | `(ℓ_i,)` | Encoded token IDs of sentence $i$ |
| Dataset matrix | $X$ | `(N, L_max)` | Padded `float32` tensor; `batch_first=True` |

Where $N$ = number of sentences, $\ell_i$ = token count of sentence $i$, $L_{\max} = \max_i \ell_i$.

---

## 3. Vectorized Implementation & Numerical Stability

1. **Where loops are unavoidable:**
   - Vocabulary construction and per-sentence encoding are inherently sequential string operations — there is no tensor-level equivalent for `sentence.split()`. The only genuine vectorized/optimized op is the final `nn.utils.rnn.pad_sequence`, which performs the rectangularization in C++.
   - Avoid `inplace`/manual padding loops (`for ...: tensor += [0]`) and let `pad_sequence` handle it.

2. **Correctness invariants:**
   - **ID range:** every real token maps to $[1, |V|]$; ID `0` is exclusively padding.
   - **Order preservation:** encoding walks `sentence.split()` in order — no sorting of tokens.
   - **Determinism:** `sorted(set(...))` yields a stable, reproducible vocabulary, so IDs are identical across runs (essential for train/test consistency).
   - **Batch order:** `positive + negative` guarantees positive sentences occupy the first rows.

3. **Numerical-stability notes:**
   - Token IDs as `float32` are exact for vocabulary sizes up to $2^{24}$, far beyond any realistic corpus — no precision concerns.
   - Later stages (embedding lookup) require integer-typed indices; converting to `long`/`int64` happens inside `nn.Embedding`, so keeping the dataset `float32` here is safe.

---

## 4. Complexity Analysis

Let $N$ = number of sentences, $T$ = total tokens across the corpus, $V$ = vocabulary size, $L_{\max}$ = longest sentence length.

- **Time Complexity:** $O(T + V \log V + N \cdot L_{\max})$
  - Building the vocabulary scans every token once: $O(T)$; sorting the unique tokens costs $O(V \log V)$; encoding re-scans each token: $O(T)$; padding writes the full $N \times L_{\max}$ matrix: $O(N \cdot L_{\max})$.
- **Space Complexity / Activation Memory:** $O(N \cdot L_{\max} + V)$
  - The padded matrix dominates ($N \cdot L_{\max}$ floats), plus the vocabulary dictionary and the intermediate per-sentence tensors. No gradients or cached activations — this is a pure data-preparation step.

---

## 5. Edge Cases & Gotchas

- [ ] **Reserved padding index:** vocabulary IDs start at `1` (`enumerate(vocab, start=1)` equivalent); never map a real token to `0`, or padding becomes ambiguous.
- [ ] **Sentence ordering:** `positive + negative` — positives first. Swapping the lists silently corrupts any label alignment downstream.
- [ ] **Variable sentence lengths:** `pad_sequence` requires a non-empty list of 1D tensors and pads all to `L_max` at the end (right padding).
- [ ] **Declared dtype:** return `float32` to match `TensorType[float]`; using `int64` would fail the contract.
- [ ] **Case & punctuation preserved:** `split()` treats `"Hello,"` and `"hello,"` as distinct tokens — the problem intentionally performs no lowercasing or normalization.
- [ ] **Ragged / empty sentences:** an empty string produces a length-0 tensor; `pad_sequence` still pads it to $L_{\max}$ with zeros.
- [ ] Common pitfalls during implementation:
  - Forgetting `batch_first=True`, which yields a transposed `(L_max, N)` tensor.
  - Rebuilding the vocabulary per-sentence instead of once over the combined corpus (produces inconsistent IDs).

---

## 6. Clean Code

```python
import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        sentences = positive + negative

        vocab = sorted(set(
            word
            for sentence in sentences
            for word in sentence.split()
        ))
        word_to_id = {word: i + 1 for i, word in enumerate(vocab)}

        tensors = [
            torch.tensor(
                [word_to_id[word] for word in sentence.split()],
                dtype=torch.float32
            )
            for sentence in sentences
        ]

        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Visual Example

```text
positive = ["good film", "i love it"]
negative = ["bad plot"]

Combined corpus (positive first):
   s0 = "good film"   -> {"good":1, "film":3}
   s1 = "i love it"   -> {"i":2, "love":4, "it":5}
   s2 = "bad plot"    -> {"bad":0? no -> "bad" gets an ID too}

sorted vocab: [bad, film, good, i, it, love, plot]
word_to_id:   bad=1, film=2, good=3, i=4, it=5, love=6, plot=7
              (ID 0 is reserved for padding)

Encoded (variable length):
   s0 -> [3, 2]           len 2
   s1 -> [4, 6, 5]        len 3
   s2 -> [1, 7]           len 2

pad_sequence(batch_first=True) -> shape (3, 3):
   [ 3, 2, 0 ]   <- s0 padded with 0 at the end
   [ 4, 6, 5 ]   <- s1 (longest, untouched)
   [ 1, 7, 0 ]   <- s2 padded
```

---

### Architectural Pipeline

```text
  positive[] + negative[]           (order matters: positives first)
            ↓
  split() every sentence -> unique tokens
            ↓
  sorted(set(...)) -> vocabulary
            ↓
  {word: id}, ids from 1            (0 reserved for padding)
            ↓
  encode each sentence -> list of variable-length tensors
            ↓
  nn.utils.rnn.pad_sequence(..., batch_first=True)
            ↓
  Padded float32 matrix X ∈ R^(N × L_max)
```

* **Next Review Date:**
* **Key Takeaway:** Text becomes a tensor in three deterministic steps — build a sorted vocabulary (IDs starting at `1`, leaving `0` for padding), encode sentences while preserving word order, then let `pad_sequence(batch_first=True)` rectangularize the ragged batch into a `(N, L_max)` float matrix.

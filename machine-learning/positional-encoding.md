# Positional Encoding

- **Problem Link:** [NeetCode - Positional Encoding](https://neetcode.io/problems/positional-encoding)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `NLP`
- **Implementation Framework:** `NumPy`
- **Core Concept / Formulation:** `Sinusoidal Positional Encoding, Absolute Position Signals, Multi-Frequency Sinusoids, Broadcasting`
- **Last Practiced:** 2026-10-24
- **Proficiency Level:**
  - [ ] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Derivations

### Overview & Motivation
Self-attention is **permutation-equivariant**: if the input tokens are shuffled and the attention weights are shuffled the same way, the output is identical. The mechanism has no built-in notion of order, so *"dog bites man"* and *"man bites dog"* would produce the same set of contextualized vectors. Positional encoding injects an order signal into the token representations before they enter the Transformer.

The original Transformer (Vaswani et al., 2017) uses a fixed, parameter-free **sinusoidal** encoding. For position $pos$ and feature index $j$:

$$
PE(pos, j) =
\begin{cases}
\sin\left(\dfrac{pos}{10000^{2i / d_{\text{model}}}}\right), & j = 2i \text{ (even)} \\[6pt]
\cos\left(\dfrac{pos}{10000^{2i / d_{\text{model}}}}\right), & j = 2i + 1 \text{ (odd)}
\end{cases}
$$

where $i = 0, 1, \dots, \lceil d_{\text{model}}/2 \rceil - 1$ indexes a **frequency pair**. A single angle matrix is sufficient:

$$
\text{angles}(pos, i) = pos \cdot \underbrace{\frac{1}{10000^{2i / d_{\text{model}}}}}_{\text{angle\_rate}(i)}
$$

$$
PE(pos, 0{:}{:}2) = \sin(\text{angles}), \qquad
PE(pos, 1{:}{:}2) = \cos(\text{angles})
$$

The even column $2i$ and the odd column $2i+1$ share the **same angle**, so the encoding stores a sine/cosine pair per frequency. This is why only $d_{\text{model}}/2$ distinct frequencies are needed.

### Why Sinusoids?
- **Bounded, smooth values:** every entry lies in $[-1, 1]$, so the encoding cannot dominate the token embedding it is added to.
- **Geometric frequency ladder:** the wavelength grows as $2\pi \cdot 10000^{2i/d_{\text{model}}}$, giving both fast-changing (local) and slow-changing (global) dimensions.
- **Relative-position algebra:** for any fixed shift $k$, $PE(pos + k)$ can be written as a **linear transformation** of $PE(pos)$ (angle-addition identities). The model can therefore learn to attend by relative offset.
- **Extrapolation:** the formula is defined for any $pos$, including positions longer than those seen during training; no lookup table caps the sequence length.
- **No parameters:** nothing to train, nothing to initialize, and no risk of overfitting position embeddings.

### How It Is Used
The encoding is **added element-wise** to the token embeddings, not concatenated:

$$
X_{\text{input}} = E[\text{token\_ids}] \cdot \sqrt{d_{\text{model}}} + PE
$$

The scaling factor $\sqrt{d_{\text{model}}}$ is applied to the embeddings (in the original paper) so that the positional signal remains a meaningful fraction of the total input magnitude.

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Positions | $pos$ | `(seq_len, 1)` | Column vector of positions $0 \dots seq\_len-1$ |
| Dimension indices | $2i$ | `(ceil(d_model / 2),)` | Even feature indices `0, 2, 4, ...` |
| Angle rates | $\omega_i$ | `(ceil(d_model / 2),)` | $1 / 10000^{2i/d_{\text{model}}}$ per frequency |
| Angles | $\Theta$ | `(seq_len, ceil(d_model / 2))` | Outer product of positions and angle rates |
| Encoding (even cols) | $PE_{:,0::2}$ | `(seq_len, ceil(d_model / 2))` | $\sin(\Theta)$ |
| Encoding (odd cols) | $PE_{:,1::2}$ | `(seq_len, floor(d_model / 2))` | $\cos(\Theta[:, :d_{\text{model}}//2])$ |
| Output | $PE$ | `(seq_len, d_model)` | Full sinusoidal positional encoding |

Notes:
- For **even** $d_{\text{model}}$, the sine and cosine slices each have exactly $d_{\text{model}}/2$ columns and `angles[:, :d_model // 2]` is a no-op.
- For **odd** $d_{\text{model}}$, `np.arange(0, d_model, 2)` produces $\lceil d_{\text{model}}/2 \rceil$ entries, but the odd-column slice `pe[:, 1::2]` has only $\lfloor d_{\text{model}}/2 \rfloor$ columns. The slice `angles[:, :d_model // 2]` trims the surplus angle so the broadcast assignment stays shape-compatible.

---

## 3. Vectorized Implementation & Numerical Stability

1. **Broadcasting Replaces Nested Loops:**
   - A naive implementation is a double loop over `pos` and `i`, computing `sin(pos / 10000 ** (2 * i / d_model))` inside. This costs Python interpreter overhead per element and rebuilds the same denominator repeatedly.
   - Broadcasting lifts the computation into a single outer product: `positions` is `(seq_len, 1)` and `angle_rates` is `(ceil(d/2),)`, so `positions * angle_rates` yields the full `(seq_len, ceil(d/2))` angle matrix in one C-level operation. `sin` and `cos` then act on entire arrays.

2. **Strided Assignment for Interleaving:**
   - `pe[:, 0::2] = np.sin(angles)` writes every even column; `pe[:, 1::2] = np.cos(...)` writes every odd column.
   - NumPy performs this as a single strided assignment rather than a per-column Python loop, and the two slices are views, not copies.

3. **Attribute Caching:**
   - The same `angles` matrix feeds both the sine and cosine branches, so the expensive `power` and outer product are computed once per call instead of once per parity.

4. **Numerical Stability Safeguards:**
   - $\sin$ and $\cos$ are bounded on $[-1, 1]$, so there is no overflow or underflow risk regardless of `seq_len`.
   - The exponent $2i/d_{\text{model}} \in [0, 1)$ keeps $10000^{2i/d_{\text{model}}}$ in $[1, 10000)$, and `1.0 /` forces true division in float64; using integer division here would collapse the angle rates to `0` or `1`.
   - `np.zeros` defaults to `float64`, matching the declared `NDArray[np.float64]` return type.

5. **Rounding Invariant:**
   - The problem expects the encoding rounded to **5 decimal places** via `np.round(pe, 5)`.

---

## 4. Complexity Analysis

Let $S = \text{seq\_len}$ and $d = d_{\text{model}}$.

- **Time Complexity:** $O(S \cdot d)$ — Explanation: building the angle matrix is an outer product of $S \times \lceil d/2 \rceil$ entries, and applying sine and cosine touches each of the $S \cdot d$ output entries exactly once. The transcendental functions dominate the constant factor, but the asymptotic cost is linear in the number of output elements.
- **Space Complexity:** $O(S \cdot d)$ — Explanation: the returned `pe` matrix has $S \cdot d$ float64 entries, and the intermediate `angles` matrix adds another $S \cdot \lceil d/2 \rceil$ entries (about half the output size). The position vector, dimension vector, and angle rates are $O(S + d)$.

---

## 5. Edge Cases & Gotchas

- [x] **Odd `d_model`:** the even-column slice is one wider than the odd-column slice; `angles[:, :d_model // 2]` must trim the extra column before assigning cosine. Without it, the broadcast fails.
- [x] **`d_model = 1`:** `dimensions = [0]`, so the single column receives $\sin(pos)$ and the odd slice is empty. The code degrades gracefully.
- [x] **`seq_len = 0`:** `positions` is empty and the result has shape `(0, d_model)`; both strided assignments accept empty slices.
- [x] **Position 0:** $\sin(0) = 0$ and $\cos(0) = 1$, so the first row is `[0, 1, 0, 1, ...]` — useful as a quick sanity check on the interleaving order.
- [x] **Column ordering (even = sin, odd = cos):** swapping the two slices still yields a valid encoding but breaks consistency with the reference formula and with checkpoints trained on the standard layout. The *pairing* of column $2i$ and $2i+1$ on the same angle is the invariant that matters.
- [x] **Exponent base and scaling:** the denominator is $10000^{2i/d_{\text{model}}}$, **not** $10000^{i/d_{\text{model}}}$; using `dimensions` (which already steps by 2) and dividing by `d_model` implements the correct exponent.
- [x] **Integer arithmetic:** `1.0 / np.power(10000.0, ...)` and the float exponent keep the computation in floating point; integer division would destroy the frequencies.
- [x] **Rounding precision:** round only at the end, to 5 decimals, after all arithmetic — rounding intermediate values would accumulate error.
- [x] **Addition vs. concatenation:** this function returns the encoding only; in a Transformer it is summed with the token embeddings, which requires both to have shape `(seq_len, d_model)`.
- **Common pitfalls during implementation:**
  - Using `np.arange(d_model)` instead of `np.arange(0, d_model, 2)`, which halves the intended frequencies incorrectly and misaligns the angle matrix.
  - Forgetting the column-vector reshape `[:, np.newaxis]`, which makes `positions * angle_rates` an invalid or unintended broadcast.
  - Assuming `d_model` is always even and skipping the cosine trim.

---

## 6. Clean Code

```python
import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        positions = np.arange(seq_len)[:, np.newaxis]
        dimensions = np.arange(0, d_model, 2)

        angle_rates = 1.0 / np.power(10000.0, dimensions / d_model)
        angles = positions * angle_rates

        pe = np.zeros((seq_len, d_model))

        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

        return np.round(pe, 5)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Visual Computation Example ($seq\_len = 4$, $d_{\text{model}} = 4$)

```text
Step 1 — frequencies (one per pair):
  dimensions  = [0, 2]
  exponents   = [0/4, 2/4] = [0.0, 0.5]
  angle_rates = [1/10000^0, 1/10000^0.5] = [1.0, 0.01]

Step 2 — outer product angles = positions * angle_rates (4 x 2):
             i=0      i=1
  pos 0  [  0.00,    0.00 ]
  pos 1  [  1.00,    0.01 ]
  pos 2  [  2.00,    0.02 ]
  pos 3  [  3.00,    0.03 ]

Step 3 — interleave sin into even columns, cos into odd columns:
             j=0      j=1      j=2      j=3
           sin(a0)  cos(a0)  sin(a1)  cos(a1)
  pos 0  [  0.000,   1.000,   0.000,   1.000 ]
  pos 1  [  0.841,   0.540,   0.010,   1.000 ]
  pos 2  [  0.909,  -0.416,   0.020,   1.000 ]
  pos 3  [  0.141,  -0.990,   0.030,   1.000 ]
```

Observation: the first pair (`i = 0`) oscillates quickly with position, while the second pair (`i = 1`) changes slowly. Stacking many such frequencies gives every position a unique, smoothly varying fingerprint.

---

### Architectural Pipeline

```text
        seq_len, d_model
               ↓
   positions (seq_len, 1)  ×  angle_rates (ceil(d/2),)
               ↓
      angles (seq_len, ceil(d/2))
               ↓
   ┌───────────┴────────────┐
   ↓                        ↓
 sin(angles)            cos(angles[:, :d//2])
   ↓                        ↓
 pe[:, 0::2]            pe[:, 1::2]
   └───────────┬────────────┘
               ↓
      PE (seq_len, d_model)
               ↓
   round to 5 decimals  →  add to token embeddings
```

* **Next Review Date:**
* **Key Takeaway:** Sinusoidal positional encoding is a parameter-free outer product between positions and a geometric ladder of frequencies; writing sine into even columns and cosine into odd columns gives every position a unique, bounded, and extrapolatable fingerprint that self-attention can use to reason about order.

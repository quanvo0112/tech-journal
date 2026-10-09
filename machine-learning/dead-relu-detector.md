# Dead ReLU Detector

- **Problem Link:** [NeetCode - Dead ReLU Detector](https://neetcode.io/problems/dead-relu-detector)
- **Difficulty:** `Medium`
- **NeetCode ML Module:** `Training`
- **Implementation Framework:** `PyTorch`
- **Core Concept / Formulation:** `Dead ReLU Detection, PyTorch Forward Hooks (register_forward_hook), Batch-Wise Inactivity Reduction ((activations == 0).all(dim=0)), Multi-Priority Remediation Decision Tree (use_leaky_relu, reinitialize, reduce_learning_rate, healthy)`
- **Last Practiced:** 2026-10-09
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)
  - [ ] Level 2: Struggled / Non-vectorized / Shape or dimension mismatch bugs
  - [ ] Level 3: Needed editorial or mathematical derivation hints

---

## 1. Mathematical Formulation & Diagnostic Logic

### The Dying ReLU Problem
The standard Rectified Linear Unit (ReLU) activation function is defined as:
$$a = \text{ReLU}(z) = \max(0, z)$$

Its derivative with respect to pre-activation $z$ is:
$$\frac{\partial a}{\partial z} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

When an update step knocks a neuron's weights into a regime where $z = W x + b \le 0$ for all input samples in the training distribution:
1. The neuron's output is identically zero: $a = 0$.
2. The local gradient collapses to zero: $\frac{\partial a}{\partial z} = 0$.
3. Through backpropagation, the gradient received by its incoming weights vanishes:
   $$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot x^T = 0$$

Once this happens, gradient descent can never update $W$ or $b$ again. The neuron is permanently "dead".

---

### Batch-Wise Inactivity Criterion

In practice with mini-batch training, a neuron $j \in [1, D]$ is defined as **dead** for an evaluation batch $X \in \mathbb{R}^{B \times D_{\text{in}}}$ if and only if its activation output is exactly $0$ across **all** $B$ samples:

$$\text{is\_dead}(j) = \bigwedge_{i=1}^B (a_{i, j} = 0)$$

The dead neuron fraction $\rho_{\text{dead}}$ for a layer with $D$ output channels is:
$$\rho_{\text{dead}} = \frac{1}{D} \sum_{j=1}^D \mathbb{I}\left[\forall i \in [1, B], \, a_{i, j} = 0\right]$$

Implemented in PyTorch across batch dimension $0$:
```python
dead = (activations == 0).all(dim=0)
dead_fraction = dead.float().mean().item()
```

> **Crucial Distinction:** A neuron that outputs $0$ on some samples but positive values on others is merely sparse, **not dead**. It continues to receive gradients on samples where $z > 0$. Hence, testing `(activations == 0).all(dim=0)` is required rather than `(activations == 0).mean()`.

---

### Priority-Ordered Remediation Decision Hierarchy

The `suggest_fix(dead_fractions)` function evaluates network health following a strict precedence order. The first satisfied condition immediately dictates the suggested remediation:

| Priority | Evaluation Condition | Action / Remediation | Root Cause & Justification |
|:---:|:---|:---:|:---|
| **1** | $\exists l: \rho_{\text{dead}}^{(l)} > 0.5$ | `use_leaky_relu` | Over 50% capacity loss in a layer. Switching to Leaky ReLU ($\max(\alpha z, z)$ with $\alpha = 0.01$) guarantees a non-zero gradient $\alpha$ for negative inputs, reviving dead units. |
| **2** | $\rho_{\text{dead}}^{(1)} > 0.3$ | `reinitialize` | Early layer failure suffocates subsequent representations. Typically caused by poor initial weight variance or large negative initial biases. Re-initialization (e.g. He/Kaiming normal) restores signal flow. |
| **3** | $\rho_{\text{dead}}^{(1)} < \rho_{\text{dead}}^{(2)} < \dots < \rho_{\text{dead}}^{(L)}$ ($L > 1$) and $\rho_{\text{dead}}^{(L)} > 0.1$ | `reduce_learning_rate` | Cascading death: dead neuron ratio grows strictly monotonically with depth and terminates above 10%. Overly large learning rates cause aggressive updates that progressively push deeper layers into negative saturation. |
| **4–5** | All other cases | `healthy` | Dead neuron ratios remain within tolerable operating thresholds. |

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input Evaluation Batch | $x$ | `(B, D_in)` | Evaluation data passed into the model |
| Raw Hook Activation | `output` | `(B, D)` or `(D,)` or `(B, C, H, W)` | Tensor captured by `register_forward_hook` |
| Normalized Activation | `activations` | `(B, D_features)` | Reshaped to 2D matrix (batch, features) |
| Dead Indicator Vector | `dead` | `(D_features,)` | Boolean vector where `True` indicates inactive across batch |
| Layer Dead Fraction | $\rho_{\text{dead}}$ | Scalar `float` | Proportion of dead neurons in layer, rounded to 4 decimals |
| Diagnostic Sequence | `dead_fractions` | `List[float]` of length $L$ | Ordered dead fractions for all $L$ ReLU layers |

---

## 3. Vectorized Implementation & Numerical Stability

### 1. PyTorch Forward Hooks (`register_forward_hook`)
Instead of modifying the forward function of custom models or subclassing modules, `module.register_forward_hook(hook)` inspects activations at runtime:
- `output.detach()` decouples the captured activation from PyTorch's computation graph, preventing unnecessary graph retention and memory leaks.
- `with torch.no_grad():` ensures zero autograd overhead during the forward pass.
- `try ... finally` block guarantees every handle in `handles` is cleanly dereferenced via `handle.remove()`, preventing hook accumulation across repeated invocations.

### 2. Multi-Dimensional Tensor Normalization
Activation outputs can vary in shape depending on layer type (e.g., 1D for single-sample vectors, 2D for linear layers, 4D for convolutional layers):
- If `activations.ndim == 1`: unsqueeze to `(1, D)` so `dim=0` correctly represents the single batch item.
- If `activations.ndim >= 2`: flatten spatial/feature dimensions to `(B, -1)`.

### 3. Strict Monotonicity Check
Priority 3 requires strict increase ($<$), not non-decreasing ($\le$):
```python
all(dead_fractions[i] < dead_fractions[i + 1] for i in range(len(dead_fractions) - 1))
```

---

## 4. Complexity Analysis

- **Time Complexity:** $O(\text{FLOPs}(M) + \sum_{l=1}^L B \cdot D_l)$
  - Executing `model(x)` takes the standard forward inference time of the network.
  - For each of the $L$ ReLU layers, checking `(activations == 0).all(dim=0)` performs a single pass over $B \times D_l$ elements.
  - Suggesting fixes takes $O(L)$ to inspect the list of length $L$.
- **Space Complexity / Activation Memory:** $O(B \cdot \max_{l} D_l)$
  - Detached activations are processed layer-by-layer inside the hook callback and immediately reduced to a scalar float, requiring only transient memory for the largest layer's activations.

---

## 5. Edge Cases & Gotchas

- [x] **Single-Sample Input (`ndim == 1`):** A 1D tensor `(D,)` would cause `.all(dim=0)` to reduce across features instead of the batch. Calling `activations.unsqueeze(0)` preserves the batch dimension.
- [x] **Spatial Activations (`ndim == 4`):** For convolutional activations `(B, C, H, W)`, reshaping to `(B, -1)` flattens channels and spatial locations into individual neuron activations.
- [x] **Partial Inactivity is NOT Dead:** A neuron outputting zero on 99% of samples but positive on 1% is active and updates via gradients. Using `.all(dim=0)` correctly flags only 100% inactive neurons.
- [x] **Resource Cleanup with `try ... finally`:** Failure to remove hook handles leaves memory dangling and causes duplicate callbacks on subsequent model calls.
- [x] **Single-Layer Models in Priority 3:** When $L \le 1$, strict depth-wise increase is undefined; the guard `len(dead_fractions) > 1` prevents false positives.

---

## 6. Clean Code

```python
import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        handles = []

        def hook(module, inputs, output):
            activations = output.detach()

            if activations.ndim == 1:
                activations = activations.unsqueeze(0)
            else:
                activations = activations.reshape(activations.shape[0], -1)

            dead = (activations == 0).all(dim=0)
            dead_fraction = dead.float().mean().item()

            dead_fractions.append(round(dead_fraction, 4))

        for module in model.modules():
            if isinstance(module, nn.ReLU):
                handles.append(module.register_forward_hook(hook))

        try:
            with torch.no_grad():
                model(x)
        finally:
            for handle in handles:
                handle.remove()

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Priority 1: Any layer has dead fraction > 0.5
        if any(fraction > 0.5 for fraction in dead_fractions):
            return "use_leaky_relu"

        # Priority 2: First layer has dead fraction > 0.3
        if dead_fractions and dead_fractions[0] > 0.3:
            return "reinitialize"

        # Priority 3: Strictly increasing with depth,
        # and the last layer has dead fraction > 0.1
        if (
            len(dead_fractions) > 1
            and all(
                dead_fractions[i] < dead_fractions[i + 1]
                for i in range(len(dead_fractions) - 1)
            )
            and dead_fractions[-1] > 0.1
        ):
            return "reduce_learning_rate"

        # Priorities 4 and 5 both return healthy
        return "healthy"
```

---

## 7. Architecture Walkthrough & Diagnostic Pipeline

```text
       Forward Pass with Registered Hooks
       ----------------------------------
                 Input Batch x
                      │
                 ┌────▼────┐
                 │ Layer 1 │
                 └────┬────┘
                      │
               ┌──────▼──────┐
               │   nn.ReLU   │ ──► [Hook] activations (B, D1)
               └──────┬──────┘            │
                      │                   ▼
                 ┌────▼────┐      (act == 0).all(dim=0) ──► dead_frac_1
                 │ Layer 2 │
                 └────┬────┘
                      │
               ┌──────▼──────┐
               │   nn.ReLU   │ ──► [Hook] activations (B, D2)
               └──────┬──────┘            │
                      │                   ▼
                      ▼           (act == 0).all(dim=0) ──► dead_frac_2
                     ...

       Remediation Priority Evaluation
       -------------------------------
         dead_fractions = [ρ1, ρ2, ...]
                      │
     Any ρ_i > 0.5?  ───► YES ───► "use_leaky_relu"
            │ NO
     ρ_1 > 0.3?      ───► YES ───► "reinitialize"
            │ NO
     Strictly increasing with depth
     AND last layer ρ_L > 0.1?
            │
           YES ───► "reduce_learning_rate"
            │ NO
            ▼
        "healthy"
```

* **Next Review Date:** 2026-10-16
* **Key Takeaway:** A neuron is dead only when inactive across every batch sample (`(activations == 0).all(dim=0)`); forward hooks allow modular, zero-overhead diagnostic telemetry without altering network architecture.


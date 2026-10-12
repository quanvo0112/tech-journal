# Handwritten Digit Classifier

- **Problem Link:** [NeetCode - Digit Classifier](https://neetcode.io/problems/digit-classifier)
- **Difficulty:** `Easy`
- **NeetCode ML Module:** `Training`
- **Implementation Framework:** `PyTorch`
- **Core Concept / Formulation:** `Feedforward Neural Network, Multi-Layer Perceptron (MLP), Dropout Regularization, Sigmoid Output`
- **Last Practiced:** 2026-10-12
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, vectorized, optimal)

---

## 1. Mathematical Formulation & Derivations

### Overview & Objective
Build a feedforward neural network (Multi-Layer Perceptron) for classifying handwritten digits (0-9) from 784-pixel flattened images. The architecture follows a standard 2-layer MLP with ReLU activation, Dropout regularization, and Sigmoid output layer.

### Architecture Specification
- **Input Layer:** 784 features (flattened 28×28 MNIST images)
- **Hidden Layer:** Linear(784, 512) → ReLU → Dropout(p=0.2)
- **Output Layer:** Linear(512, 10) → Sigmoid → 10 outputs in (0, 1)

### Key Equations

**Hidden Layer (Linear + ReLU + Dropout):**
$$\mathbf{h} = \text{ReLU}(\mathbf{X} \mathbf{W}_1^T + \mathbf{b}_1)$$
$$\tilde{\mathbf{h}} = \text{Dropout}_{p=0.2}(\mathbf{h})$$

**Output Layer (Linear + Sigmoid):**
$$\mathbf{z} = \tilde{\mathbf{h}} \mathbf{W}_2^T + \mathbf{b}_2$$
$$\hat{\mathbf{y}} = \sigma(\mathbf{z}) = \frac{1}{1 + e^{-\mathbf{z}}} \in (0, 1)^{10}$$

Where:
- $\mathbf{X} \in \mathbb{R}^{B \times 784}$ is the input batch
- $\mathbf{W}_1 \in \mathbb{R}^{512 \times 784}$, $\mathbf{b}_1 \in \mathbb{R}^{512}$
- $\mathbf{W}_2 \in \mathbb{R}^{10 \times 512}$, $\mathbf{b}_2 \in \mathbb{R}^{10}$

---

## 2. Tensor Shapes & Dimension Architecture

| Variable / Tensor | Symbol | Shape / Dimensions | Description |
|:---:|:---:|:---:|:---|
| Input Batch | $\mathbf{X}$ | `(B, 784)` | Flattened images, B samples |
| Hidden Weight | $\mathbf{W}_1$ | `(512, 784)` | First linear layer weights |
| Hidden Bias | $\mathbf{b}_1$ | `(512,)` | First linear layer bias |
| Hidden Pre-Act | $\mathbf{X}\mathbf{W}_1^T + \mathbf{b}_1$ | `(B, 512)` | Linear projection |
| Hidden Activation | $\mathbf{h}$ | `(B, 512)` | ReLU applied element-wise |
| Hidden w/ Dropout | $\tilde{\mathbf{h}}$ | `(B, 512)` | Dropout during training |
| Output Weight | $\mathbf{W}_2$ | `(10, 512)` | Second linear layer weights |
| Output Bias | $\mathbf{b}_2$ | `(10,)` | Second linear layer bias |
| Output Logits | $\mathbf{z}$ | `(B, 10)` | Linear projection to 10 classes |
| Predictions | $\hat{\mathbf{y}}$ | `(B, 10)` | Sigmoid outputs in (0, 1) |

---

## 3. Vectorized Implementation & Numerical Stability

1. **Sequential Module Composition:**
   - `nn.Sequential` chains layers in order: Linear → ReLU → Dropout → Linear → Sigmoid
   - All operations are fully vectorized; no explicit loops over batch or feature dimensions

2. **Dropout Behavior:**
   - During **training** (`model.train()`): randomly zeros 20% of activations, scales remaining by $1/(1-p) = 1.25$
   - During **evaluation** (`model.eval()`): disabled entirely, acts as identity
   - Seed controlled via `torch.manual_seed(0)` for reproducibility

3. **Sigmoid Output:**
   - Maps each logit independently to $(0, 1)$
   - Does not enforce sum-to-1 (unlike Softmax); appropriate for multi-label or independent probability interpretation

4. **Precision Rounding:**
   - `torch.round(output, decimals=4)` maintains tensor type and applies element-wise rounding
   - Avoids conversion to Python list, preserves autograd graph if needed

---

## 4. Complexity Analysis

Let $B$ = batch size, $D_{in} = 784$, $H = 512$, $D_{out} = 10$:

- **Time Complexity (Forward Pass):** $\mathcal{O}(B \cdot D_{in} \cdot H + B \cdot H \cdot D_{out})$
  - First linear layer: $B \times 784 \times 512$ MACs
  - Second linear layer: $B \times 512 \times 10$ MACs
  - Dominated by first layer: $\mathcal{O}(B \cdot 401,408)$
- **Space Complexity (Activations):** $\mathcal{O}(B \cdot H + B \cdot D_{out})$
  - Hidden activations: $(B, 512)$
  - Output logits: $(B, 10)$
  - Parameter storage (fixed): $\approx 784 \times 512 + 512 \times 10 \approx 402,000$ parameters

---

## 5. Edge Cases & Gotchas

- [x] **Seed Consistency:** `torch.manual_seed(0)` in both `__init__` (weight init) and `forward` (dropout randomness)
- [x] **Dropout Mode:** Ensure `model.train()` / `model.eval()` is called appropriately by the test harness
- [x] **Rounding:** Use `torch.round(tensor, decimals=4)` not Python `round()` to preserve tensor type
- [x] **Sigmoid vs Softmax:** Architecture specifies Sigmoid; do not substitute Softmax
- [x] **Layer Order:** Must follow: Linear → ReLU → Dropout → Linear → Sigmoid exactly

---

## 6. Clean Code

```python
import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
            nn.Sigmoid()
        )

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        output = self.model(images)
        return torch.round(output, decimals=4)
```

---

## 7. Architecture Walkthrough & Key Takeaways

### Feedforward Flow

```text
Input (B, 784)
       │
       ▼
┌──────────────────┐
│  Linear(784,512) │  ← Learnable: W₁(512,784), b₁(512)
└────────┬─────────┘
         ▼
    ┌────────┐
    │ ReLU   │  ← Element-wise max(0, x)
    └────┬───┘
         ▼
┌──────────────────┐
│ Dropout(p=0.2)   │  ← Training: zero 20%, scale 1.25
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Linear(512,10)  │  ← Learnable: W₂(10,512), b₂(10)
└────────┬─────────┘
         ▼
    ┌────────┐
    │Sigmoid │  ← 1/(1+exp(-z)) per element
    └────┬───┘
         ▼
Output (B, 10)  ← Rounded to 4 decimals
```

* **Next Review Date:** 
* **Key Takeaway:** PyTorch's `nn.Sequential` cleanly composes standard building blocks (Linear, ReLU, Dropout, Sigmoid). `torch.manual_seed(0)` in both `__init__` and `forward` ensures full reproducibility of weight initialization and dropout stochasticity. The Sigmoid output gives independent per-class probabilities rather than a normalized distribution.
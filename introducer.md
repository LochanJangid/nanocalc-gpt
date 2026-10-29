# Engineering Notes & Design Rationale

This document captures the original problem-solving log and intuitive breakthroughs behind **nanoCalc-GPT**. The core philosophy: **force the model to discover generalizable algorithmic patterns rather than memorize input-output pairs.**

---

## 1. The Right-to-Left Carry Problem

### Problem
When training an autoregressive transformer on standard arithmetic expressions (e.g., `11+06=17`), learning fails to converge to algorithmic accuracy:
- **Human arithmetic executes right-to-left** (starting from the least significant digit to propagate carries).
- **Causal transformers generate left-to-right**.

In standard format `11+06=17`, the model must output `1` (the tens digit) before it computes `7` (the ones digit). Because the tens digit depends on whether the ones digit generated a carry, the model is forced to predict high-order digits without the required context, creating an information bottleneck.

### Decision & Solution
Keep the prompt (LHS) in natural reading order (`11+06=`), but **reverse the target digits on the RHS** (`+7100` for 17):
- The model outputs the units digit first (`7`), followed by the tens digit (`1`), and padded zeroes (`00`).
- Each token decision is conditioned on the already-generated lower-order digits and their carries.
- Result: Addition and subtraction achieve **100.0% exact-match accuracy**.

*(Note: As documented in the main README, division is the exception — long division is naturally MSB-first, so division digits remain unreversed.)*

---

## 2. Loss Masking: Answer-Only Supervision

### Problem
Calculating cross-entropy loss across the entire sequence (operands + operator + answer) dilutes gradient signal:
- The prompt tokens (`11`, `+`, `06`, `=`) are randomly generated inputs with no learnable internal pattern.
- Penalizing the model for failing to predict random prompt tokens forces network capacity into modeling input noise.

### Decision & Solution
**Compute loss strictly on the answer slice** (`logits[:, 5:11]` against `targets[:, 5:11]`):
- The model focuses 100% of its representation capacity on the functional mapping from LHS to RHS.
- Prevents gradient dispersion and accelerates convergence.

---

## 3. Generalization vs. Memorization

### Insight
Arithmetic is fundamentally algorithmic, not statistical. A model that achieves low training loss by acting as a lookup table will fail on held-out pairs.
- By applying **weight decay ($0.1$)** and **cosine learning rate scheduling**, we heavily penalize over-parameterized memorization.
- By enforcing **fixed 12-character token sequences**, positional embeddings map consistently to syntactic roles (operand 1, operator, operand 2, equals, sign, digits).

---

## Intuition
When children learn addition, teachers demonstrate the pattern and rules explicitly. Here, the transformer receives only raw input-output pairs and an optimization signal — and reconstructs the mechanics of carrying independently through multi-head self-attention.

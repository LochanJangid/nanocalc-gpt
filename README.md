# nanoCalc-GPT 🧮

> **I taught a tiny GPT to do arithmetic from scratch — no formulas, no calculator, just examples and a loss function.**

**Best result: 99.51% exact-match accuracy** on held-out `+ - * /` problems.  
Model: decoder-only transformer · 1.2M parameters · trains in ~3 min on a single GPU.

---

## The core question

GPT models predict the next token left-to-right. Arithmetic is also solved left-to-right — *except carrying*, which propagates right-to-left. Does a transformer discover this on its own, from examples alone?

Short answer: **yes, with one deliberate hint on the data format.**

---

## Key design decisions

### 1 — Reverse the answer digits (for `+ - *`)

A standard `11+06=17` forces the model to write `1` (the tens digit) before it knows the carry from `7+6=13`. There's no stable left-to-right pattern.

**Fix:** store the answer least-significant digit first.

```
11+06=+7100    (answer is 17, stored as "7100" reversed)
20-83=-3600    (answer is -63, stored as "3600" reversed)
```

Now each digit only depends on digits already generated. Addition went from ~50% to **100%**.

### 2 — Don't reverse division

Division is long division: you compute the *most* significant digit first, then work down. Reversing it would force the model to write the hardest digit first.

```
84/06=+0014    (84 ÷ 6 = 14, stored as-is — not reversed)
```

Keeping division in natural order improved division accuracy from 80.70% → **85.96%**.

### 3 — Fixed-width, fixed-length examples

Every example is exactly 12 characters: `AA{op}BB={S}DDDD\n`

```
43*57=+2451\n    ← 6 chars prompt + 5 chars answer + 1 newline = 12
```

No padding guesswork. Every position always means the same thing.

### 4 — Mask the loss to the answer only

The operands (`43`, `57`) and operator (`*`) are random. Scoring the model on predicting them adds noise and slows learning. Loss is computed **only on positions 6–10** (the answer tokens).

### 5 — Weight decay against memorization

Multiplication has 10,000 unique training pairs — large enough to memorize. Adding `weight_decay=0.1` to AdamW pushed the model toward the general rule. Multiplication accuracy went from ~62% (memorized) to **99.30%** (generalized).

---

## Architecture

```
Input tokens  →  Token Embedding (vocab=16, dim=128)
              +  Positional Embedding (context=12, dim=128)
              →  6× Transformer Block:
                    LayerNorm → Multi-Head Attention (4 heads, causal mask)
                    LayerNorm → Feed-Forward (128 → 512 → 128, ReLU)
              →  LayerNorm → Linear → logits (vocab=16)
```

| Hyperparameter | Value |
|---|---|
| Embedding dim (`n_emb`) | 128 |
| Transformer blocks | 6 |
| Attention heads | 4 |
| Head size | 32 |
| Context length | 12 |
| Vocabulary size | 16 |
| **Total parameters** | **1,195,536** |
| Batch size | 128 |
| Steps | 30,000 |
| Optimizer | AdamW (weight_decay=0.1) |
| Scheduler | OneCycleLR (cosine, 3% warmup) |
| Learning rate | 1e-3 |

---

## Results

Evaluated on 3,057 held-out examples the model never saw during training.  
**Exact match** = every answer token must be correct (no partial credit).

| Operator | Test examples | Exact match |
|---|---|---|
| `+` Addition | 1,000 | **100.00%** |
| `-` Subtraction | 1,000 | **100.00%** |
| `*` Multiplication | 1,000 | **99.30%** |
| `/` Division | 57 | **85.96%** |
| **Overall** | **3,057** | **99.51%** |

*(Results from Colab GPU run, `n_emb=128`, `n_blocks=6`. Local CPU checkpoint: 97.55%.)*

### Ablation: architecture vs. accuracy

| `n_emb` / `n_blocks` | `*` accuracy | `/` accuracy | Overall |
|---|---|---|---|
| 64 / 4 | 96.00% | 78.95% | 98.30% |
| 64 / 6 | 81.30% | 82.46% | 93.56% |
| 128 / 4 | 96.60% | 84.21% | 98.59% |
| **128 / 6** | **99.30%** | **85.96%** | **99.51%** |

> Six blocks at width 64 hurt. Deeper + narrow is harder to train, not smarter.  
> Width and depth together unlock the best result.

---

## Project structure

```
nanocalc-gpt/
├── data/
│   ├── train.txt          # 36,000 training examples (balanced across operators)
│   └── test.txt           # 3,057 held-out test examples
├── artifacts/
│   └── model.pt           # Saved checkpoint (model weights + config dict)
├── model.py               # GPTConfig, GPT, Block, Attention, FeedForward
├── dataset.py             # Tokenizer, vocab, get_data() sampler
├── generate_data.py       # Generates data/train.txt and data/test.txt
├── train.py               # Local training script (CPU/GPU)
├── evaluate.py            # CLI evaluation: per-operator accuracy table
├── exact_match.py         # Core evaluation library (exact_match, predict, load_saved_model)
├── practice.ipynb         # Interactive evaluation notebook (load model, run predictions)
├── train_colab.ipynb      # Self-contained Colab training notebook (GPU)
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/LochanJangid/nanocalc-gpt
cd nanocalc-gpt
pip install -r requirements.txt

# Generate dataset
python generate_data.py

# Evaluate the saved checkpoint
python evaluate.py --split test
python evaluate.py --split test --show-errors     # show wrong predictions
python evaluate.py --split test --operator "*"    # filter one operator

# Train from scratch (CPU; ~hours — use Colab for GPU)
python train.py
```

### Single prediction (Python)

```python
from exact_match import load_saved_model, predict

model, config = load_saved_model("artifacts/model.pt")

print(predict(model, "37*14="))   # {'prompt': '37*14=', 'raw_prediction': '+8510', 'math_value': 518}
print(predict(model, "84/06="))   # {'prompt': '84/06=', 'raw_prediction': '+0014', 'math_value': 14}
print(predict(model, "99+01="))   # {'prompt': '99+01=', 'raw_prediction': '+0010', 'math_value': 100}
```

### Train in Google Colab (recommended)

Open `train_colab.ipynb` in Colab. It:
1. Auto-clones the repo and sets up the environment
2. Generates the dataset with the correct format
3. Trains the model (~3 min on a T4 GPU)
4. Evaluates exact-match accuracy
5. Downloads `artifacts/model.pt` to your machine automatically

---

## Honest limitations

- **Operand range is 00–99.** The model has never seen 3-digit numbers. It almost certainly fails on them — that's not a bug, it's a scope boundary.
- **Division has only 57 test examples** (only pairs where `a % b == 0` are included). One wrong answer is worth ~1.75%, so treat those numbers carefully.
- **Results are from a single training run.** Multi-seed variance analysis hasn't been done yet.
- **This isn't a proof of generalization.** It shows the model handles unseen *pairs within the 00–99 range*, not that it learned the full algorithm.

---

## What I want to try next

- Scratchpad for multiplication: write out partial products before the answer
- Test on 3-digit operands with a model trained only on 2-digit ones
- Ablation study confirming each trick (reversal, loss masking, weight decay) independently

---

## Built on

- Andrej Karpathy's *[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)*

---

Made by **Lochan Jangid** · [LinkedIn](https://linkedin.com/in/lochan-jangid) · [Portfolio](https://lochan.vercel.app)

*If you spot something wrong or know a better approach, open an issue — I read all of them.*

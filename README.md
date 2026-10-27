# nanocalc-gpt

A lightweight GPT model built in PyTorch to learn and perform 2-digit arithmetic calculations (+, -, *, /).

## Project Structure

```
nanocalc-gpt/
├── data/                  # Structured data directory
│   ├── train.txt          # Generated raw text training equations
│   ├── test.txt           # Generated raw text testing equations
│   ├── train.pt           # Pre-encoded training tensor
│   └── test.pt            # Pre-encoded testing tensor
├── plots/                 # Saved training loss curves
│   └── loss.png
├── checkpoints/           # Saved model weights
│   └── model.pt
├── model.py               # GPT architecture (CasualMultiHeadAttention, Block, GPT)
├── dataset.py             # Vocabulary, encoding/decoding, and batch loading
├── data.py                # Re-export alias for dataset.py
├── generate_data.py       # Dataset generation script (saves text and encoded tensors)
├── train.py               # Training loop, evaluation, loss plotting, checkpointing
└── README.md
```

## Quickstart

### 1. Generate & Encode Dataset
Generate arithmetic examples and automatically encode them into `data/`:
```bash
python generate_data.py
```
Or encode an existing `train.txt` and `test.txt`:
```bash
python dataset.py
```

### 2. Train the Model
Run the decoupled training pipeline:
```bash
python train.py
```
Optional training flags:
```bash
python train.py --epochs 30000 --batch_size 128 --lr 0.001 --weight_decay 0.1
```

### 3. Check Model Architecture
Inspect parameter count and perform a forward pass sanity check:
```bash
python model.py
```

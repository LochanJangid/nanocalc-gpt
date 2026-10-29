"""
Training script for nanoCalc-GPT.
Trains a decoder-only causal transformer on arithmetic operations (+, -, *, /)
using cross-entropy loss masked to answer tokens only.
"""
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple
import torch

from dataset import vocab_size
from model import GPT, GPTConfig

# Compatibility alias for unpickling legacy checkpoints saved with __main__.Config
Config = GPTConfig


def train(
    epochs: int = 30000,
    lr: float = 1e-3,
    n_emb: int = 128,
    n_head: int = 4,
    n_blocks: int = 6,
    batch_size: int = 128,
    checkpoint_path: str = "artifacts/model.pt",
    seed: int = 13457,
) -> Tuple[GPT, GPTConfig]:
    """Train nanoCalc-GPT and save checkpoint to artifacts/model.pt."""
    torch.manual_seed(seed)

    config = GPTConfig(
        n_emb=n_emb,
        n_head=n_head,
        n_blocks=n_blocks,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        vocab_size=vocab_size,
    )

    print(f"Using device: {config.device}")
    print(f"Model configuration: {config}")

    model = GPT(config)
    model.to(config.device)
    print(f"Total model parameters: {model.get_num_params():,}")

    model.train()
    model.fit()

    artifacts_dir = Path(checkpoint_path).parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
        },
        checkpoint_path,
    )
    print(f"\nSaved model checkpoint to: {checkpoint_path}")

    from exact_match import exact_match
    print("\nEvaluating exact match accuracy on test set...")
    test_res = exact_match(model, data="test")
    print(test_res.summary())

    return model, config


if __name__ == "__main__":
    train()

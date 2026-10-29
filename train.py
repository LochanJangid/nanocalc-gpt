from dataclasses import dataclass
from pathlib import Path
import torch
from dataset import vocab_size
from model import GPT, GPTConfig


# HYPERPARAMETER CONFIG
@dataclass
class Config:
    context_size: int = 12
    batch_size: int = 128
    n_head: int = 4
    n_emb: int = 64  # must be divisible by n_head
    head_size: int = 64 // 4
    n_blocks: int = 4
    epochs: int = 30000
    lr: float = 1e-3
    eval_iter: int = 50
    eval_time: int = 30000 // 20
    vocab_size: int = vocab_size
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(13457)

config = Config()

model = GPT(config)
model.to(config.device)

if __name__ == "__main__":
    print(f"Using device: {config.device}")
    print(f"Model initialized with config: {config}")
    model.train()
    model.fit()

    # Save model state_dict to artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        model_path,
    )
    print(f"Saved model checkpoint to {model_path}")

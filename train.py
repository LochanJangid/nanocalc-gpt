from dataclasses import asdict
from pathlib import Path
import torch
from dataset import vocab_size
from model import GPT, GPTConfig


# Train using GPTConfig from model.py directly (single source of truth).
# Override defaults here as needed before constructing the config object.
_N_EMB = 128     # embedding dimension; must be divisible by n_head
_N_HEAD = 4
_N_BLOCKS = 6
_EPOCHS = 30000
_LR = 1e-3


torch.manual_seed(13457)

config = GPTConfig(
    n_emb=_N_EMB,
    n_head=_N_HEAD,
    n_blocks=_N_BLOCKS,
    epochs=_EPOCHS,
    lr=_LR,
    vocab_size=vocab_size,
)

model = GPT(config)
model.to(config.device)

if __name__ == "__main__":
    print(f"Using device: {config.device}")
    print(f"Model initialized with config: {config}")
    model.train()
    model.fit()

    # Save model state_dict to artifacts with plain dict config
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
        },
        model_path,
    )
    print(f"Saved model checkpoint to {model_path}")

    # Evaluate exact match accuracy
    from exact_match import exact_match
    print("\nEvaluating exact match accuracy on test set...")
    test_res = exact_match(model, data="test")
    print(test_res.summary())

import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import DEFAULT_DATA_DIR, get_batch, load_dataset, vocab_size
from model import GPT

# Default Hyperparameters
CONTEXT_SIZE = 12
BATCH_SIZE = 128
N_HEAD = 4
N_EMB = 64
EPOCHS = 30000
LR = 1e-3
EVAL_ITER = 50
WEIGHT_DECAY = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(13457)


@torch.no_grad()
def evaluation(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    eval_iter: int = EVAL_ITER,
    batch_size: int = BATCH_SIZE,
    context_size: int = CONTEXT_SIZE,
    device: str = DEVICE,
):
    model.eval()
    train_lossi = []
    test_lossi = []
    for _ in range(eval_iter):
        x, y = get_batch("train", train_data=train_data, batch_size=batch_size, context_size=context_size, device=device)
        logits = model(x)
        train_loss = F.cross_entropy(logits[:, 5:11].reshape(-1, vocab_size), y[:, 5:11].reshape(-1))

        x, y = get_batch("test", test_data=test_data, batch_size=batch_size, context_size=context_size, device=device)
        logits = model(x)
        test_loss = F.cross_entropy(logits[:, 5:11].reshape(-1, vocab_size), y[:, 5:11].reshape(-1))

        train_lossi.append(train_loss)
        test_lossi.append(test_loss)
    model.train()
    return torch.tensor(train_lossi).mean(), torch.tensor(test_lossi).mean()


def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    eval_iter: int = EVAL_ITER,
    data_dir: Path = DEFAULT_DATA_DIR,
    device: str = DEVICE,
    save_plot: bool = True,
    save_checkpoint: bool = True,
):
    print(f"Using device: {device}")
    train_data, test_data = load_dataset(data_dir=data_dir)
    print(f"Loaded dataset from {data_dir}: {len(train_data)} train tokens, {len(test_data)} test tokens")

    model = GPT(
        vocab_size=vocab_size,
        context_size=CONTEXT_SIZE,
        n_emb=N_EMB,
        n_head=N_HEAD,
        n_blocks=4,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, maximize=False, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=epochs,
        pct_start=0.03,
        anneal_strategy="cos",
    )

    eval_time = max(1, epochs // 20)
    train_lossi = []
    test_lossi = []

    start = datetime.now()
    print(f"Starting training for {epochs} epochs...")

    for i in range(epochs):
        # Forward pass
        x, y = get_batch("train", train_data=train_data, batch_size=batch_size, context_size=CONTEXT_SIZE, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits[:, 5:11].reshape(-1, vocab_size), y[:, 5:11].reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluate periodically
        if i % eval_time == 0:
            train_loss, test_loss = evaluation(
                model=model,
                train_data=train_data,
                test_data=test_data,
                eval_iter=eval_iter,
                batch_size=batch_size,
                context_size=CONTEXT_SIZE,
                device=device,
            )
            train_lossi.append(train_loss.item())
            test_lossi.append(test_loss.item())
            print(f"{i} Train: {train_loss:.4f} Test: {test_loss:.4f}")

    end = datetime.now()
    duration = end - start
    print(f"Training completed in: {duration}")

    if save_plot:
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / "loss.png"

        plt.figure(figsize=(10, 6))
        plt.plot(train_lossi, label="Train Loss")
        plt.plot(test_lossi, label="Test Loss")
        plt.title(f"Duration: {duration}")
        plt.xlabel(f"Evaluation Step (x{eval_time})")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        # Also maintain loss.png at root for backward compatibility
        plt.savefig("loss.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Loss plot saved to {plot_path} and loss.png")

    if save_checkpoint:
        checkpoints_dir = Path("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        ckpt_path = checkpoints_dir / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": vocab_size,
                "n_emb": N_EMB,
                "context_size": CONTEXT_SIZE,
                "n_head": N_HEAD,
                "epochs": epochs,
            },
            ckpt_path,
        )
        print(f"Model checkpoint saved to {ckpt_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train nanocalc-gpt")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Data directory")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        data_dir=Path(args.data_dir),
    )

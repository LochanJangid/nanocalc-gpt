from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import get_data, vocab_size


@dataclass
class GPTConfig:
    context_size: int = 12
    vocab_size: int = vocab_size
    n_emb: int = 128
    n_head: int = 4
    head_size: Optional[int] = None
    n_blocks: int = 6
    batch_size: int = 128
    epochs: int = 30000
    lr: float = 1e-3
    eval_iter: int = 50
    eval_time: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.head_size is None:
            self.head_size = self.n_emb // self.n_head
        if self.eval_time is None:
            self.eval_time = max(1, self.epochs // 20)


@torch.no_grad()
def evaluation(model, config=None):
    if config is None:
        config = getattr(model, "config", None)
    model.eval()
    train_lossi = []
    test_lossi = []
    eval_iter = getattr(config, "eval_iter", 50)
    v_size = getattr(config, "vocab_size", vocab_size)
    batch_size = getattr(config, "batch_size", 128)
    context_size = getattr(config, "context_size", 12)
    device = getattr(config, "device", next(model.parameters()).device)

    for _ in range(eval_iter):
        x, y = get_data("train", device=device, batch_size=batch_size, context_size=context_size)
        logits = model(x)
        train_loss = F.cross_entropy(logits[:, 5:11].reshape(-1, v_size), y[:, 5:11].reshape(-1))

        x, y = get_data("test", device=device, batch_size=batch_size, context_size=context_size)
        logits = model(x)
        test_loss = F.cross_entropy(logits[:, 5:11].reshape(-1, v_size), y[:, 5:11].reshape(-1))

        train_lossi.append(train_loss.item())
        test_lossi.append(test_loss.item())

    model.train()
    return torch.tensor(train_lossi).mean(), torch.tensor(test_lossi).mean()


class CasualMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = getattr(config, "head_size", config.n_emb // config.n_head)
        self.query = nn.Linear(config.n_emb, config.n_emb)
        self.key = nn.Linear(config.n_emb, config.n_emb)
        self.value = nn.Linear(config.n_emb, config.n_emb)
        self.register_buffer("mask", torch.tril(torch.ones(config.context_size, config.context_size)))
        self.proj = nn.Linear(config.n_emb, config.n_emb)

    def forward(self, x):
        # x: B, T, C
        B, T, C = x.shape
        # B, T, C -> B, T, n_head, head_size -> B, n_head, T, head_size
        Q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        V = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        scores = (Q @ k.transpose(2, 3)) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        bow = F.softmax(scores, dim=-1)
        out = bow @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_emb
        self.ff = nn.Sequential(
            nn.Linear(config.n_emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.n_emb),
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.mh = CasualMultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_emb)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.mh(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = GPTConfig(**kwargs)
        elif isinstance(config, dict):
            config = GPTConfig(**config)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_emb)
        self.positional_embedding = nn.Embedding(config.context_size, config.n_emb)
        n_blocks = getattr(config, "n_blocks", 4)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(n_blocks)])
        self.l_norm = nn.LayerNorm(config.n_emb)
        self.ln_layer = nn.Linear(config.n_emb, config.vocab_size)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(0, x.shape[1], device=x.device))
        tok_pos = token_emb + pos_emb
        out = self.l_norm(self.blocks(tok_pos))
        return self.ln_layer(out)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=5):
        """
        Takes conditioning token indices idx of shape (B, T) and generates
        max_new_tokens greedily using autoregressive decoding.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.context_size else idx[:, -self.config.context_size:]
            logits = self(idx_cond)
            idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_num_params(self, non_embedding=False):
        """
        Returns the number of parameters in the model.
        If non_embedding=True, excludes token and positional embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.positional_embedding.weight.numel()
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def fit(self, epochs=None, lr=None):
        cfg = self.config
        epochs = epochs or getattr(cfg, "epochs", 30000)
        lr = lr or getattr(cfg, "lr", 1e-3)
        eval_time = getattr(cfg, "eval_time", max(1, epochs // 20))
        device = getattr(cfg, "device", next(self.parameters()).device)
        v_size = getattr(cfg, "vocab_size", vocab_size)
        batch_size = getattr(cfg, "batch_size", 128)
        context_size = getattr(cfg, "context_size", 12)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, maximize=False, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=epochs, pct_start=0.03, anneal_strategy="cos"
        )
        train_lossi = []
        test_lossi = []
        start = datetime.now()
        for i in range(epochs):
            # Forward pass
            x, y = get_data("train", device=device, batch_size=batch_size, context_size=context_size)
            logits = self(x)
            loss = F.cross_entropy(logits[:, 5:11].reshape(-1, v_size), y[:, 5:11].reshape(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Evaluate
            if i % eval_time == 0:
                train_loss, test_loss = evaluation(self, cfg)
                train_lossi.append(train_loss.item())
                test_lossi.append(test_loss.item())
                print(f"{i} Train: {train_loss:.4f} Test: {test_loss:.4f}")

        end = datetime.now()
        duration = end - start
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plt.figure()
        plt.plot(train_lossi, label="Train Loss")
        plt.plot(test_lossi, label="Test Loss")
        plt.title(f"Duration: {duration}")
        plt.legend()
        plt.savefig(plots_dir / "loss.png", dpi=300, bbox_inches="tight")
        plt.savefig("loss.png", dpi=300, bbox_inches="tight")
        plt.close()


# Lazy import exact match utilities
def exact_match(*args, **kwargs):
    from exact_match import exact_match as _exact_match
    return _exact_match(*args, **kwargs)


def exact_match_by_operator(*args, **kwargs):
    from exact_match import exact_match_by_operator as _exact_match_by_operator
    return _exact_match_by_operator(*args, **kwargs)


def operator_accuracy(*args, **kwargs):
    from exact_match import operator_accuracy as _operator_accuracy
    return _operator_accuracy(*args, **kwargs)


def load_saved_model(*args, **kwargs):
    from exact_match import load_saved_model as _load_saved_model
    return _load_saved_model(*args, **kwargs)


if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPT(cfg)
    model.to(cfg.device)
    dummy_x = torch.randint(0, cfg.vocab_size, (2, cfg.context_size), device=cfg.device)
    print("Sanity check forward output shape:", model(dummy_x).shape)

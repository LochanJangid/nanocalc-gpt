import torch
import torch.nn as nn
import torch.nn.functional as F


class CasualMultiHeadAttention(nn.Module):
    def __init__(self, n_emb: int = 64, context_size: int = 12, n_head: int = 4):
        super().__init__()
        assert n_emb % n_head == 0, f"n_emb ({n_emb}) must be divisible by n_head ({n_head})"
        self.n_emb = n_emb
        self.n_head = n_head
        self.head_size = n_emb // n_head
        self.context_size = context_size

        self.query = nn.Linear(n_emb, n_emb)
        self.key = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)
        self.register_buffer("mask", torch.tril(torch.ones(context_size, context_size)))
        self.proj = nn.Linear(n_emb, n_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        Q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        V = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scores = Q @ k.transpose(2, 3)
        scores = scores / (self.head_size**0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        bow = F.softmax(scores, dim=-1)
        out = bow @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# Alias standard spelling while preserving original naming
CausalMultiHeadAttention = CasualMultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, n_emb: int = 64):
        super().__init__()
        hidden = 4 * n_emb
        self.ff = nn.Sequential(
            nn.Linear(n_emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_emb),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, n_emb: int = 64, context_size: int = 12, n_head: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_emb)
        self.mh = CasualMultiHeadAttention(n_emb=n_emb, context_size=context_size, n_head=n_head)
        self.ln2 = nn.LayerNorm(n_emb)
        self.ff = FeedForward(n_emb=n_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mh(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16,
        context_size: int = 12,
        n_emb: int = 64,
        n_head: int = 4,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.n_emb = n_emb
        self.n_head = n_head
        self.n_blocks = n_blocks

        self.token_embedding = nn.Embedding(vocab_size, n_emb)
        self.positional_embedding = nn.Embedding(context_size, n_emb)
        self.blocks = nn.Sequential(
            *[Block(n_emb=n_emb, context_size=context_size, n_head=n_head) for _ in range(n_blocks)]
        )
        self.l_norm = nn.LayerNorm(n_emb)
        self.ln_layer = nn.Linear(n_emb, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        device = x.device
        B, T = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(0, T, device=device))
        tok_pos = token_emb + pos_emb
        out = self.l_norm(self.blocks(tok_pos))
        return self.ln_layer(out)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate up to max_new_tokens autoregressively from prompt tokens idx of shape (B, T).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    print("Testing GPT model architecture...")
    model = GPT()
    dummy_input = torch.randint(0, 16, (2, 12))
    logits = model(dummy_input)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"GPT Architecture initialized successfully!")
    print(f"Total parameters: {num_params:,}")
    print(f"Input shape: {dummy_input.shape} -> Output logits shape: {logits.shape}")
    print("To train the model, run `python train.py`.")

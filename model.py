from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

chars = list("0123456789+=\n")
stoi = {s: i for i,s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
encode = lambda x: torch.tensor([stoi[s] for s in x])
decode = lambda x: "".join([itos[itensor.item()] for itensor in x])

train = [ch for example in open("train.txt").readlines() for ch in example.strip() + "\n"]
train = encode(train)
test = [ch for example in open("test.txt").readlines() for ch in example.strip() + "\n"]
test = encode(test)

# HYPERPARAMETERS
context_size=10
batch_size=64
n_head=4
n_emb=64 # it should be devidable by n_head
head_size=n_emb//n_head
epochs=10000
lr=0.001
eval_iter=300
eval_time=epochs//20
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(13457)

def get_data(data="train"):
    data = train if data=="train" else test
    rand_idx = torch.randint(0, len(data)//10-1, (batch_size, )) * 10
    X = torch.stack([data[i:i+context_size] for i in rand_idx])
    y = torch.stack([data[i+1:i+1+context_size] for i in rand_idx])

    return X.to(device), y.to(device)

@torch.no_grad()
def evaluation(model):
    model.eval()
    train_lossi = []
    test_lossi = []
    for i in range(eval_iter):
        x, y = get_data("train")
        logits = model(x)
        train_loss = F.cross_entropy(logits[:, 5:9].reshape(-1, vocab_size), y[:, 5:9].reshape(-1))
        x, y = get_data("test")
        logits = model(x)
        test_loss = F.cross_entropy(logits[:, 5:9].reshape(-1, vocab_size), y[:, 5:9].reshape(-1))
        train_lossi.append(train_loss)
        test_lossi.append(test_loss)
    model.train()
    return torch.tensor(train_lossi).mean(), torch.tensor(test_lossi).mean()

class CasualMultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(n_emb, n_emb)
        self.key = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)
        self.register_buffer("mask", torch.tril(torch.ones(context_size, context_size)))
        self.proj = nn.Linear(n_emb, n_emb)

    def forward(self, x):
        # x: B, T, C
        B, T, C = x.shape
        # B, T, C -> B, T, n_head, head_size ## --> B, n_head, T, head_size
        Q = self.query(x).view(B, T, n_head, head_size).transpose(1,2)
        k = self.key(x).view(B, T, n_head, head_size).transpose(1,2)
        V = self.value(x).view(B, T, n_head, head_size).transpose(1,2)
        scores = Q @ k.transpose(2, 3) # becaueze matrix multiplication see only last 2 dim
        scores = scores/head_size**0.5
        scores = scores.masked_fill(self.mask[:T,:T] == 0, float("-inf"))
        bow = F.softmax(scores, dim=-1)
        out = bow @ V
        out = out.transpose(1, 2)
        out = out.contiguous().view(B, T, C)
        return self.proj(out)
        
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 4*n_emb
        self.ff = nn.Sequential(nn.Linear(n_emb, hidden), nn.ReLU(), nn.Linear(hidden, n_emb))

    def forward(self, x):
        return self.ff(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_emb)
        self.mh = CasualMultiHeadAttention()
        self.ln2 = nn.LayerNorm(n_emb)
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.mh(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_emb)
        self.positional_embedding = nn.Embedding(context_size, n_emb)
        self.blocks = nn.Sequential(Block(), Block(),  Block())
        self.l_norm = nn.LayerNorm(n_emb)
        self.ln_layer = nn.Linear(n_emb, vocab_size)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(0, x.shape[1], device=device))
        tok_pos = token_emb + pos_emb
        out = self.l_norm(self.blocks(tok_pos))
        return self.ln_layer(out)

    def fit(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, maximize=False)
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs, pct_start=0.03, anneal_strategy="cos")
        train_lossi = []
        test_lossi = []
        start = datetime.now()
        for i in range(epochs):
            # Forward pass :|
            x, y = get_data("train")
            logits = self(x)
            loss = F.cross_entropy(logits[:, 5:9].reshape(-1, vocab_size), y[:, 5:9].reshape(-1))
            # Backward Pass :\
            optimizer.zero_grad()
            loss.backward()
            # update parameters :)
            optimizer.step()
            schedular.step() 
            # Evaluate
            if i%eval_time == 0:
                train_loss, test_loss = evaluation(self)
                train_lossi.append(train_loss.item())
                test_lossi.append(test_loss.item())
                print(f"{i} Train: {train_loss} Test: {test_loss}")
        end = datetime.now()
        duration = end-start
        plt.plot(train_lossi)
        plt.plot(test_lossi)
        plt.title(f"Duration: {duration}")
        plt.savefig("loss.png", dpi=300, bbox_inches="tight")

model = GPT()
model.to(device)
model.fit()
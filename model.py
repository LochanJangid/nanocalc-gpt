import torch
import torch.nn as nn
import torch.nn.functional as F

data = [ch for example in open("data1.txt").readlines() for ch in example.strip()]
chars = list("0123456789+=\n")
stoi = {s: i for i,s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
encode = lambda x: torch.tensor([stoi[s] for s in x])
decode = lambda x: "".join([itos[itensor.item()] for itensor in x])
tokend_data = encode(data)

# HYPERPARAMETERS
context_size=10
block_size=10
n_head=4
n_emb=32 # it should be devidable by n_head
head_size=n_emb//n_head

def get_data():
    rand_idx = torch.randint(0, len(tokend_data)-context_size, (block_size, ))
    X = torch.stack([tokend_data[i:i+context_size] for i in rand_idx])
    y = torch.stack([tokend_data[i+1:i+1+context_size] for i in rand_idx])

    return X, y

# token embedding
# position encoding
# Block Xn
    # layer norm two way one is direct second step by step
    # casual attention 
    #   multi head attention
    # layer norm 
    # feed forward

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
        

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_emb)
        self.positional_embedding = nn.Embedding(context_size, n_emb)
        self.mh = CasualMultiHeadAttention()

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(0, context_size))
        tok_pos = token_emb + pos_emb
        out = self.mh(tok_pos)
        return out.shape

model = GPT()
X, y = get_data()
print(model.forward(X))
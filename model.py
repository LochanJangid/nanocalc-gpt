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
n_emb=23

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

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_emb)
        self.positional_embedding = nn.Embedding(context_size, n_emb)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(0, context_size))
        tok_pos = token_emb + pos_emb

        return tok_pos.shape

model = GPT()
X, y = get_data()
print(model.forward(X))
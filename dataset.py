import torch

chars = list("0123456789+*-/=\n")
stoi = {s: i for i,s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
encode = lambda x: torch.tensor([stoi[s] for s in x])
decode = lambda x: "".join([itos[itensor.item()] for itensor in x])

train = [ch for example in open("data/train.txt").readlines() for ch in example.strip() + "\n"]
train = encode(train)

test = [ch for example in open("data/test.txt").readlines() for ch in example.strip() + "\n"]
test = encode(test)

def get_data(data="train", device="cpu", batch_size=20, context_size=12):

    data = train if data=="train" else test
    rand_idx = torch.randint(0, len(data)//12-1, (batch_size, )) * 12
    X = torch.stack([data[i:i+context_size] for i in rand_idx])
    y = torch.stack([data[i+1:i+1+context_size] for i in rand_idx])

    return X.to(device), y.to(device)
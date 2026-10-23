import random

WIDTH = 2                    # digits per operand
N = 10 ** WIDTH
TEST_FRAC = 0.1
random.seed(1337)

def fmt(a, b):
    ans = str(a + b).zfill(WIDTH + 1)[::-1]          # pad to WIDTH+1 digits, reverse
    return f"{a:0{WIDTH}d}+{b:0{WIDTH}d}={ans}"

pairs = [(a, b) for a in range(N) for b in range(N)]  # every pair exactly once
random.shuffle(pairs)

n_test = int(len(pairs) * TEST_FRAC)
splits = {"test": pairs[:n_test], "train": pairs[n_test:]}

for name, ps in splits.items():
    with open(f"{name}.txt", "w") as f:
        f.write("\n".join(fmt(a, b) for a, b in ps) + "\n")
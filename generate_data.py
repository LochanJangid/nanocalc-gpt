import random

with open("data1.txt", "w") as f:
    for i in range(99):
        i = random.randint(0, 99)
        j = random.randint(0, 99)
        f.write(f"{i}+{j}={i+j}\n")
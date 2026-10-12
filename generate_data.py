with open("data1.txt", "w") as f:
    for i in range(99):
        for j in range(99):
            f.write(f"{i}+{j}={i+j}\n")
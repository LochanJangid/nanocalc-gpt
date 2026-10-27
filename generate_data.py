from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, List, Tuple
from dataset import DEFAULT_DATA_DIR, prepare_encoded_data

SEED, TEST_FRAC, N = 1337, 0.1, 100


def result(a: int, op: str, b: int):
    return {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else None}[op]


def fmt(a: int, op: str, b: int) -> str:
    r = result(a, op, b)
    sign = "-" if r < 0 else "+"
    return f"{a:02d}{op}{b:02d}={sign}{str(abs(r)).zfill(4)[::-1]}\n"


def decode(line: str) -> int:
    """Answer of a formatted line as an int (for checks)."""
    ans = line.strip().split("=")[1]
    v = int(ans[1:][::-1])
    return -v if ans[0] == "-" else v


def all_examples() -> Dict[str, List[Tuple[int, str, int]]]:
    ex = defaultdict(list)
    for a in range(N):
        for b in range(N):
            ex["+"].append((a, "+", b))
            ex["-"].append((a, "-", b))
            ex["*"].append((a, "*", b))
            if b != 0 and a % b == 0:
                ex["/"].append((a, "/", b))
    return ex


def generate_dataset(data_dir: Path = DEFAULT_DATA_DIR, encode_data: bool = True):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    train, test = [], []
    by_op_train = {}
    for op, items in all_examples().items():
        random.shuffle(items)
        n_test = int(len(items) * TEST_FRAC)
        test += items[:n_test]
        by_op_train[op] = items[n_test:]
        print(f"{op}: {len(items)} unique  ->  train {len(items) - n_test}, test {n_test}")

    target = max(len(v) for v in by_op_train.values())
    for op, items in by_op_train.items():
        reps = items * (target // len(items)) + items[: target % len(items)]  # oversample rare ops
        train += reps

    random.shuffle(train)
    random.shuffle(test)

    for name, ps in (("train", train), ("test", test)):
        lines = [fmt(*e) for e in ps]
        assert all(decode(l) == result(a, op, b) for l, (a, op, b) in zip(lines, ps))
        assert all(len(l) == 12 for l in lines)
        out_path = data_dir / f"{name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        print(f"{out_path}: {len(lines)} examples")

    if encode_data:
        prepare_encoded_data(data_dir=data_dir, force=True)


if __name__ == "__main__":
    generate_dataset()

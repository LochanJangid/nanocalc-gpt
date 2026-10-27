import os
from pathlib import Path
from typing import Tuple, Union
import torch

# Vocabulary definition
chars = list("0123456789+*-/=\n")
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"


def encode(text: Union[str, list]) -> torch.Tensor:
    """Encode characters to a 1D torch.LongTensor of token indices."""
    return torch.tensor([stoi[s] for s in text], dtype=torch.long)


def decode(tokens: Union[torch.Tensor, list]) -> str:
    """Decode a sequence of token indices back to string."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return "".join([itos[int(t)] for t in tokens])


def prepare_encoded_data(data_dir: Union[str, Path] = DEFAULT_DATA_DIR, force: bool = False) -> Tuple[Path, Path]:
    """
    Reads train.txt and test.txt, encodes them, and saves train.pt and test.pt into data_dir.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_pt = data_dir / "train.pt"
    test_pt = data_dir / "test.pt"

    if not force and train_pt.exists() and test_pt.exists():
        print(f"Encoded dataset files already exist in {data_dir}.")
        return train_pt, test_pt

    train_txt = data_dir / "train.txt"
    test_txt = data_dir / "test.txt"

    # Fallback to base directory if text files are not yet in data_dir
    if not train_txt.exists() and (BASE_DIR / "train.txt").exists():
        train_txt = BASE_DIR / "train.txt"
    if not test_txt.exists() and (BASE_DIR / "test.txt").exists():
        test_txt = BASE_DIR / "test.txt"

    if not train_txt.exists() or not test_txt.exists():
        raise FileNotFoundError(
            f"Dataset text files not found in {data_dir} or {BASE_DIR}. "
            f"Run `python generate_data.py` first."
        )

    print(f"Encoding {train_txt}...")
    with open(train_txt, "r", encoding="utf-8") as f:
        train_chars = [ch for line in f.readlines() for ch in line.strip() + "\n"]
    train_tensor = encode(train_chars)
    torch.save(train_tensor, train_pt)
    print(f"Saved {train_pt} ({len(train_tensor)} tokens)")

    print(f"Encoding {test_txt}...")
    with open(test_txt, "r", encoding="utf-8") as f:
        test_chars = [ch for line in f.readlines() for ch in line.strip() + "\n"]
    test_tensor = encode(test_chars)
    torch.save(test_tensor, test_pt)
    print(f"Saved {test_pt} ({len(test_tensor)} tokens)")

    return train_pt, test_pt


def load_dataset(data_dir: Union[str, Path] = DEFAULT_DATA_DIR) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads encoded train and test tensors. If .pt files do not exist,
    prepares them from train.txt and test.txt automatically.
    """
    data_dir = Path(data_dir)
    train_pt = data_dir / "train.pt"
    test_pt = data_dir / "test.pt"

    if not train_pt.exists() or not test_pt.exists():
        prepare_encoded_data(data_dir)

    train_data = torch.load(train_pt, weights_only=True)
    test_data = torch.load(test_pt, weights_only=True)
    return train_data, test_data


def get_batch(
    split: str = "train",
    train_data: torch.Tensor = None,
    test_data: torch.Tensor = None,
    batch_size: int = 128,
    context_size: int = 12,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples random batch (X, y) aligned to context_size (12-char equation boundaries).
    """
    data = train_data if split == "train" else test_data
    if data is None:
        raise ValueError(f"Data for split '{split}' must be provided.")

    num_equations = len(data) // context_size
    rand_idx = torch.randint(0, num_equations - 1, (batch_size,)) * context_size
    X = torch.stack([data[i : i + context_size] for i in rand_idx])
    y = torch.stack([data[i + 1 : i + 1 + context_size] for i in rand_idx])

    return X.to(device), y.to(device)


# Alias for backward compatibility with model.py get_data
get_data = get_batch


class CalculatorDataset:
    """Convenience wrapper for loading and batching dataset."""
    def __init__(self, data_dir: Union[str, Path] = DEFAULT_DATA_DIR, device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.device = device
        self.train_data, self.test_data = load_dataset(self.data_dir)

    def get_batch(self, split: str = "train", batch_size: int = 128, context_size: int = 12):
        return get_batch(
            split=split,
            train_data=self.train_data,
            test_data=self.test_data,
            batch_size=batch_size,
            context_size=context_size,
            device=self.device,
        )


if __name__ == "__main__":
    prepare_encoded_data(force=True)

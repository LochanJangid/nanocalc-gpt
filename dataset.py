"""
dataset.py — Tokenisation and data-loading utilities for nanoCalc-GPT.

Vocabulary
----------
The model operates over a closed, 16-character vocabulary that covers every
symbol needed to express arithmetic expressions and their results::

    0 1 2 3 4 5 6 7 8 9 + * - / = \\n

Each character is mapped to a unique integer token id (see ``stoi`` / ``itos``).

Token encoding
--------------
Text is encoded character-by-character: each character in the input string is
looked up in ``stoi`` and replaced with its integer index.  Decoding is the
inverse operation.

Data loading
------------
``train.txt`` and ``test.txt`` are read **at import time** from relative paths
(``data/train.txt`` and ``data/test.txt``).  The working directory must therefore
be the project root when this module is imported.  Each file is expected to
contain one arithmetic expression per line (e.g. ``3+5=8``); lines are joined
with a ``\\n`` separator and encoded into a flat 1-D ``torch.LongTensor``.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: All characters recognised by the model, in index order.
chars: list[str] = list("0123456789+*-/=\n")

#: Mapping from character to integer token id.
stoi: dict[str, int] = {s: i for i, s in enumerate(chars)}

#: Mapping from integer token id back to character.
itos: dict[int, str] = {i: s for s, i in stoi.items()}

#: Total number of distinct tokens (== 16).
vocab_size: int = len(itos)


# ---------------------------------------------------------------------------
# Encoding / decoding helpers
# ---------------------------------------------------------------------------


def encode(text: list[str]) -> torch.Tensor:
    """Encode a sequence of characters into a 1-D integer tensor.

    Each character in *text* is looked up in :data:`stoi` and converted to its
    corresponding token id.  All characters must belong to the vocabulary.

    Args:
        text: A list of single characters to encode.

    Returns:
        A 1-D ``torch.LongTensor`` of token ids with the same length as *text*.

    Raises:
        KeyError: If any character in *text* is not present in the vocabulary.
    """
    return torch.tensor([stoi[s] for s in text])


def decode(tokens: torch.Tensor) -> str:
    """Decode a 1-D integer tensor back into a plain string.

    Each token id in *tokens* is looked up in :data:`itos` and converted to its
    corresponding character.  All token ids must be valid indices into the
    vocabulary.

    Args:
        tokens: A 1-D ``torch.LongTensor`` of token ids to decode.

    Returns:
        The reconstructed string, one character per token.

    Raises:
        KeyError: If any value in *tokens* is not a valid token id.
    """
    return "".join([itos[itensor.item()] for itensor in tokens])


# ---------------------------------------------------------------------------
# Dataset tensors (loaded at import time)
# ---------------------------------------------------------------------------

#: Flat encoded tensor for the training split.
train: torch.Tensor = encode(
    [ch for example in open("data/train.txt").readlines() for ch in example.strip() + "\n"]
)

#: Flat encoded tensor for the test split.
test: torch.Tensor = encode(
    [ch for example in open("data/test.txt").readlines() for ch in example.strip() + "\n"]
)


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------


def get_data(
    data: Literal["train", "test"] = "train",
    device: str = "cpu",
    batch_size: int = 20,
    context_size: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequence pairs.

    Sequences are sampled at aligned boundaries (multiples of 12) so that each
    sample begins at the start of a complete expression.  Targets are the
    inputs shifted one position to the right, matching the standard
    next-token-prediction objective.

    Args:
        data: Which split to sample from — ``"train"`` or ``"test"``.
        device: The PyTorch device string (e.g. ``"cpu"``, ``"cuda"``) to
            which the returned tensors are moved.
        batch_size: Number of sequences in the returned batch.
        context_size: Length (in tokens) of each sequence.

    Returns:
        A tuple ``(X, y)`` where:

        * **X** — shape ``(batch_size, context_size)``, the input token ids.
        * **y** — shape ``(batch_size, context_size)``, the target token ids
          (each row is *X* shifted one position to the right).
    """
    dataset: torch.Tensor = train if data == "train" else test
    rand_idx: torch.Tensor = torch.randint(0, len(dataset) // 12 - 1, (batch_size,)) * 12
    X: torch.Tensor = torch.stack([dataset[i : i + context_size] for i in rand_idx])
    y: torch.Tensor = torch.stack([dataset[i + 1 : i + 1 + context_size] for i in rand_idx])
    return X.to(device), y.to(device)

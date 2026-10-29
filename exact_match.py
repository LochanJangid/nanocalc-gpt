import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from dataset import decode, encode, itos, stoi, vocab_size
from model import GPT, GPTConfig


OPERATOR_NAMES: Dict[str, str] = {
    "+": "Addition",
    "-": "Subtraction",
    "*": "Multiplication",
    "/": "Division",
}

OPERATOR_ALIASES: Dict[str, str] = {
    "+": "+", "add": "+", "addition": "+", "plus": "+",
    "-": "-", "sub": "-", "subtraction": "-", "minus": "-",
    "*": "*", "mul": "*", "multiplication": "*", "times": "*", "multiply": "*",
    "/": "/", "div": "/", "division": "/", "divide": "/", "slash": "/",
}


def normalize_operator(op: str) -> str:
    """Normalizes an operator symbol or alias (e.g. 'add' -> '+')."""
    cleaned = op.strip().lower()
    if cleaned in OPERATOR_ALIASES:
        return OPERATOR_ALIASES[cleaned]
    if op in OPERATOR_NAMES:
        return op
    raise ValueError(f"Unknown operator: {op!r}. Supported operators: {list(OPERATOR_NAMES.keys())}")


class ExactMatchResult(dict):
    """
    Container for exact match evaluation results with dictionary, attribute,
    and operator-specific inspection capabilities.
    """

    def __init__(
        self,
        accuracy: float,
        correct: int,
        total: int,
        by_operator: Optional[Dict[str, Dict[str, Any]]] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
        errors_by_operator: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        samples_by_operator: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        split_name: str = "",
        operator_filter: Optional[List[str]] = None,
    ):
        by_op = by_operator or {}
        samples_list = samples or []
        errs_by_op = errors_by_operator or {op: [] for op in OPERATOR_NAMES}
        samps_by_op = samples_by_operator or {op: [] for op in OPERATOR_NAMES}
        errors_count = total - correct

        super().__init__(
            accuracy=accuracy,
            correct=correct,
            total=total,
            errors=errors_count,
            by_operator=by_op,
            samples=samples_list,
            errors_by_operator=errs_by_op,
            samples_by_operator=samps_by_op,
            split_name=split_name,
            operator_filter=operator_filter,
        )
        self.accuracy = accuracy
        self.correct = correct
        self.total = total
        self.errors = errors_count
        self.by_operator = by_op
        self.samples = samples_list
        self.errors_by_operator = errs_by_op
        self.samples_by_operator = samps_by_op
        self.split_name = split_name
        self.operator_filter = operator_filter

    def __float__(self) -> float:
        return float(self.accuracy)

    def __format__(self, format_spec: str) -> str:
        return format(self.accuracy, format_spec)

    @property
    def operators(self) -> List[str]:
        """List of operators present in this evaluation."""
        return sorted(self.by_operator.keys())

    def op_accuracy(self, op: str) -> float:
        """Returns accuracy for a given operator symbol or name (e.g. '+' or 'addition')."""
        symbol = normalize_operator(op)
        if symbol in self.by_operator:
            return float(self.by_operator[symbol]["accuracy"])
        raise KeyError(f"Operator '{op}' ({symbol}) not found in evaluation results.")

    def op_correct(self, op: str) -> int:
        """Returns number of correct predictions for an operator."""
        symbol = normalize_operator(op)
        return int(self.by_operator.get(symbol, {}).get("correct", 0))

    def op_total(self, op: str) -> int:
        """Returns total number of examples for an operator."""
        symbol = normalize_operator(op)
        return int(self.by_operator.get(symbol, {}).get("total", 0))

    def op_errors(self, op: str) -> int:
        """Returns number of mispredictions for an operator."""
        symbol = normalize_operator(op)
        return int(self.by_operator.get(symbol, {}).get("errors", 0))

    def get_operator(self, op: str) -> Dict[str, Any]:
        """Returns full statistics dictionary for an operator."""
        symbol = normalize_operator(op)
        if symbol in self.by_operator:
            return self.by_operator[symbol]
        raise KeyError(f"Operator '{op}' ({symbol}) not found in evaluation results.")

    def get_errors(self, op: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Returns list of error samples.
        If op is provided, returns errors only for that operator.
        """
        if op is not None:
            symbol = normalize_operator(op)
            errs = self.errors_by_operator.get(symbol, [])
        else:
            errs = [s for s in self.samples if not s["correct"]]

        return errs[:limit] if limit is not None else errs

    def get_correct(self, op: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Returns list of correctly predicted samples.
        If op is provided, returns samples only for that operator.
        """
        if op is not None:
            symbol = normalize_operator(op)
            corrs = [s for s in self.samples_by_operator.get(symbol, []) if s["correct"]]
        else:
            corrs = [s for s in self.samples if s["correct"]]

        return corrs[:limit] if limit is not None else corrs

    def table(self) -> str:
        """Returns a formatted ASCII table of per-operator accuracy and overall metrics."""
        rows = []
        rows.append("+--------------------+---------+---------+--------+----------+")
        rows.append("| Operator           |   Total | Correct | Errors | Accuracy |")
        rows.append("+--------------------+---------+---------+--------+----------+")

        for op, stats in sorted(self.by_operator.items()):
            op_label = f"{op} ({stats.get('name', OPERATOR_NAMES.get(op, 'Unknown'))})"
            t = stats["total"]
            c = stats["correct"]
            e = stats["errors"]
            a = stats["accuracy"]
            rows.append(f"| {op_label:<18} | {t:>7} | {c:>7} | {e:>6} | {a:>7.2%} |")

        rows.append("+--------------------+---------+---------+--------+----------+")
        rows.append(
            f"| {'Overall':<18} | {self.total:>7} | {self.correct:>7} | {self.errors:>6} | {self.accuracy:>7.2%} |"
        )
        rows.append("+--------------------+---------+---------+--------+----------+")
        return "\n".join(rows)

    def summary(
        self,
        show_table: bool = True,
        show_errors: bool = False,
        max_errors_per_op: int = 3,
    ) -> str:
        """Generates a human-readable evaluation summary."""
        header_parts = ["Exact Match"]
        if self.split_name:
            header_parts.append(f"Split: {self.split_name}")
        if self.operator_filter:
            header_parts.append(f"Filter: {','.join(self.operator_filter)}")
        header = f"=== {' | '.join(header_parts)} ==="

        lines = [
            header,
            f"Overall Accuracy: {self.correct}/{self.total} ({self.accuracy:.4%})",
        ]

        if show_table and self.by_operator:
            lines.append("\n" + self.table())
        elif self.by_operator:
            lines.append("Per-Operator Accuracy:")
            for op, stats in sorted(self.by_operator.items()):
                c, t, a = stats["correct"], stats["total"], stats["accuracy"]
                name = stats.get("name", OPERATOR_NAMES.get(op, ""))
                lines.append(f"  Op '{op}' ({name}): {c}/{t} ({a:.4%})")

        if show_errors:
            lines.append("\n=== Error Samples by Operator ===")
            has_any_errors = False
            for op in sorted(self.by_operator.keys()):
                errs = self.get_errors(op, limit=max_errors_per_op)
                if errs:
                    has_any_errors = True
                    op_name = OPERATOR_NAMES.get(op, op)
                    lines.append(f"Operator '{op}' ({op_name}) Errors (showing up to {max_errors_per_op}):")
                    for s in errs:
                        lines.append(
                            f"  Prompt: {s['prompt']} | Pred: {s['predicted']} (decoded: {s['decoded_pred']}) "
                            f"| Target: {s['target']} (expected: {s['decoded_target']})"
                        )
            if not has_any_errors:
                lines.append("  (No errors found! 100% accuracy)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = []
        if self.split_name:
            parts.append(f"split='{self.split_name}'")
        if self.operator_filter:
            parts.append(f"filter={self.operator_filter}")
        parts.append(f"accuracy={self.accuracy:.4%}")
        parts.append(f"correct={self.correct}/{self.total}")
        return f"ExactMatchResult({', '.join(parts)})"


def load_saved_model(
    checkpoint_path: Union[str, Path] = "artifacts/model.pt",
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[GPT, GPTConfig]:
    """
    Loads a saved model checkpoint, ensuring compatibility across unpickling environments.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        fallback_path = Path("checkpoints/model.pt")
        if fallback_path.exists():
            checkpoint_path = fallback_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compatibility shim: checkpoints saved with __main__.Config in train.py
    if not hasattr(sys.modules.get("__main__"), "Config"):
        try:
            import train
            sys.modules["__main__"].Config = train.Config
        except Exception:
            sys.modules["__main__"].Config = GPTConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract configuration
    raw_cfg = checkpoint.get("config", None)
    if isinstance(raw_cfg, GPTConfig):
        cfg = raw_cfg
        cfg.device = str(device)
    elif raw_cfg is not None:
        cfg = GPTConfig(
            context_size=getattr(raw_cfg, "context_size", 12),
            vocab_size=getattr(raw_cfg, "vocab_size", vocab_size),
            n_emb=getattr(raw_cfg, "n_emb", 64),
            n_head=getattr(raw_cfg, "n_head", 4),
            head_size=getattr(raw_cfg, "head_size", None),
            n_blocks=getattr(raw_cfg, "n_blocks", 4),
            batch_size=getattr(raw_cfg, "batch_size", 128),
            lr=getattr(raw_cfg, "lr", 1e-3),
            device=str(device),
        )
    else:
        cfg = GPTConfig(
            context_size=checkpoint.get("context_size", 12),
            vocab_size=checkpoint.get("vocab_size", vocab_size),
            n_emb=checkpoint.get("n_emb", 64),
            n_head=checkpoint.get("n_head", 4),
            device=str(device),
        )

    model = GPT(cfg)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, cfg


def decode_arithmetic(prompt_str: str, ans_str: str) -> Optional[int]:
    """
    Decodes the model's reversed answer representation back into an integer.
    Example: prompt='53+00=', ans='+3500' -> +53
    """
    ans_str = ans_str.strip()
    if not ans_str or ans_str[0] not in ("+", "-"):
        return None
    try:
        sign = -1 if ans_str[0] == "-" else 1
        return sign * int(ans_str[1:][::-1])
    except Exception:
        return None


@torch.no_grad()
def predict(
    model: GPT,
    prompt: str,
    device: Optional[Union[str, torch.device]] = None,
    max_new_tokens: int = 5,
) -> Dict[str, Union[str, int, None]]:
    """
    Runs autoregressive generation on an arithmetic prompt (e.g. '12+34=' or '12+34').
    Returns raw prediction and decoded integer arithmetic result.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    prompt_clean = prompt.strip()
    if not prompt_clean.endswith("="):
        prompt_clean += "="
    if len(prompt_clean) != 6:
        raise ValueError(f"Prompt must be formatted as 2-digit op 2-digit '=', e.g. '12+34=', got: {prompt!r}")

    x = encode(prompt_clean).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        logits = model(x)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, next_tok], dim=1)

    raw_pred = decode(x[0, 6:6 + max_new_tokens])
    math_value = decode_arithmetic(prompt_clean, raw_pred)

    return {
        "prompt": prompt_clean,
        "raw_prediction": raw_pred,
        "math_value": math_value,
    }


def _prepare_data_tensor(
    data: Union[str, Path, torch.Tensor, List[str]]
) -> Tuple[torch.Tensor, str]:
    """Helper to convert data inputs into a (N, 12) token tensor."""
    split_name = ""
    if isinstance(data, str):
        split_name = data
        if data.lower() == "train":
            data_path = Path("data/train.txt")
        elif data.lower() == "test":
            data_path = Path("data/test.txt")
        else:
            data_path = Path(data)

        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                lines = [line.strip() + "\n" for line in f if line.strip()]
            chars = [ch for line in lines for ch in line]
            tensor_data = encode(chars).view(-1, 12)
            return tensor_data, split_name
        raise FileNotFoundError(f"Data file not found: {data_path}")

    elif isinstance(data, Path):
        split_name = data.stem
        with open(data, "r", encoding="utf-8") as f:
            lines = [line.strip() + "\n" for line in f if line.strip()]
        chars = [ch for line in lines for ch in line]
        tensor_data = encode(chars).view(-1, 12)
        return tensor_data, split_name

    elif isinstance(data, list):
        split_name = "custom_list"
        lines = [line.strip() + "\n" for line in data if line.strip()]
        chars = [ch for line in lines for ch in line]
        tensor_data = encode(chars).view(-1, 12)
        return tensor_data, split_name

    elif isinstance(data, torch.Tensor):
        split_name = "tensor"
        if data.ndim == 1:
            tensor_data = data.view(-1, 12)
        else:
            tensor_data = data
        return tensor_data, split_name

    raise ValueError(f"Unsupported data type for evaluation: {type(data)}")


@torch.no_grad()
def exact_match(
    model: Optional[Union[GPT, str, Path]] = None,
    data: Union[str, Path, torch.Tensor, List[str]] = "test",
    checkpoint_path: Union[str, Path] = "artifacts/model.pt",
    batch_size: int = 512,
    device: Optional[Union[str, torch.device]] = None,
    num_samples: Optional[int] = None,
    operator: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
    return_details: bool = False,
    max_recorded_errors_per_op: int = 50,
) -> Union[ExactMatchResult, Dict[str, ExactMatchResult]]:
    """
    Evaluates exact match accuracy of the saved or provided model, with rich
    per-operator metrics, error tracking, and optional operator filtering.

    For each example in the format 'a{op}b={sign}{rev_digits}\\n',
    the prompt 'a{op}b=' (length 6) is given to the model, and the model
    autoregressively generates the next 5 tokens. An exact match is achieved
    if and only if all 5 predicted tokens match the ground-truth target tokens.

    Args:
        model: GPT model instance or checkpoint path. If None, loads from checkpoint_path.
        data: 'train', 'test', 'both', a file path, a torch.Tensor, or a list of lines.
        checkpoint_path: Path to model checkpoint (used if model is None or a path).
        batch_size: Batch size for batched evaluation.
        device: Device to use for computation ('cuda', 'cpu', etc.).
        num_samples: Maximum number of samples to evaluate (evaluates all if None).
        operator: Optional operator symbol or name to filter evaluation (e.g. '+', '*',
                  ['+', '-'], 'multiplication', etc.).
        return_details: If True, stores predictions for all samples in result.samples.
        max_recorded_errors_per_op: Max number of error cases to record per operator.

    Returns:
        ExactMatchResult, or a dict of {'train': ExactMatchResult, 'test': ExactMatchResult} if data='both'.
    """
    # Handle data="both"
    if isinstance(data, str) and data.lower() in ("both", "all"):
        if model is None or isinstance(model, (str, Path)):
            resolved_ckpt = checkpoint_path if model is None else model
            loaded_model, _ = load_saved_model(resolved_ckpt, device=device)
        else:
            loaded_model = model

        train_res = exact_match(
            model=loaded_model,
            data="train",
            batch_size=batch_size,
            device=device,
            num_samples=num_samples,
            operator=operator,
            return_details=return_details,
            max_recorded_errors_per_op=max_recorded_errors_per_op,
        )
        test_res = exact_match(
            model=loaded_model,
            data="test",
            batch_size=batch_size,
            device=device,
            num_samples=num_samples,
            operator=operator,
            return_details=return_details,
            max_recorded_errors_per_op=max_recorded_errors_per_op,
        )
        return {"train": train_res, "test": test_res}

    # Resolve model
    if model is None or isinstance(model, (str, Path)):
        resolved_ckpt = checkpoint_path if model is None else model
        model_instance, _ = load_saved_model(resolved_ckpt, device=device)
    else:
        model_instance = model

    model_instance.eval()
    if device is None:
        device = next(model_instance.parameters()).device
    else:
        model_instance.to(device)

    # Prepare data tensor: shape (N, 12)
    data_tensor, split_name = _prepare_data_tensor(data)

    # Handle operator filter
    filter_symbols: Optional[List[str]] = None
    if operator is not None:
        if isinstance(operator, str):
            filter_symbols = [normalize_operator(operator)]
        else:
            filter_symbols = [normalize_operator(op) for op in operator]

        allowed_tokens = [stoi[sym] for sym in filter_symbols if sym in stoi]
        mask = torch.isin(data_tensor[:, 2], torch.tensor(allowed_tokens))
        data_tensor = data_tensor[mask]

    if num_samples is not None:
        data_tensor = data_tensor[:num_samples]

    total_samples = data_tensor.shape[0]
    if total_samples == 0:
        return ExactMatchResult(
            accuracy=0.0,
            correct=0,
            total=0,
            by_operator={},
            samples=[],
            errors_by_operator={},
            samples_by_operator={},
            split_name=split_name,
            operator_filter=filter_symbols,
        )

    total_correct = 0
    by_operator_counts: Dict[str, Dict[str, int]] = {}
    errors_by_operator: Dict[str, List[Dict[str, Any]]] = {}
    samples_by_operator: Dict[str, List[Dict[str, Any]]] = {}
    all_samples: List[Dict[str, Any]] = []

    for i in range(0, total_samples, batch_size):
        batch = data_tensor[i : i + batch_size].to(device)
        B = batch.shape[0]

        # Prompt: first 6 tokens (e.g. "53+00=")
        x = batch[:, :6]
        # Target: next 5 tokens (e.g. "+3500")
        target = batch[:, 6:11]

        # Autoregressive generation of 5 tokens
        for _ in range(5):
            idx_cond = x if x.size(1) <= getattr(model_instance.config, "context_size", 12) else x[:, -12:]
            logits = model_instance(idx_cond)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, next_tok], dim=1)

        preds = x[:, 6:11]
        matches = (preds == target).all(dim=-1)  # shape (B,)
        batch_correct = matches.sum().item()
        total_correct += batch_correct

        # Operator breakdown (operator token is at index 2 in prompt)
        op_tokens = batch[:, 2].cpu().tolist()
        matches_cpu = matches.cpu().tolist()

        for j in range(B):
            op_char = itos.get(op_tokens[j], "?")
            if op_char not in by_operator_counts:
                by_operator_counts[op_char] = {"correct": 0, "total": 0, "errors": 0}
                errors_by_operator[op_char] = []
                samples_by_operator[op_char] = []

            by_operator_counts[op_char]["total"] += 1
            is_match = matches_cpu[j]

            if is_match:
                by_operator_counts[op_char]["correct"] += 1
            else:
                by_operator_counts[op_char]["errors"] += 1

            # Build record if needed
            need_error_record = (not is_match) and (len(errors_by_operator[op_char]) < max_recorded_errors_per_op)
            need_sample_record = return_details or (len(samples_by_operator[op_char]) < 10)

            if need_error_record or need_sample_record:
                prompt_str = decode(batch[j, :6].cpu())
                target_str = decode(target[j].cpu())
                pred_str = decode(preds[j].cpu())
                dec_target = decode_arithmetic(prompt_str, target_str)
                dec_pred = decode_arithmetic(prompt_str, pred_str)

                record = {
                    "prompt": prompt_str,
                    "target": target_str,
                    "predicted": pred_str,
                    "correct": is_match,
                    "operator": op_char,
                    "operator_name": OPERATOR_NAMES.get(op_char, op_char),
                    "decoded_target": dec_target,
                    "decoded_pred": dec_pred,
                }

                if need_sample_record:
                    samples_by_operator[op_char].append(record)
                    if return_details or len(all_samples) < 50:
                        all_samples.append(record)

                if need_error_record:
                    errors_by_operator[op_char].append(record)

    accuracy = total_correct / total_samples

    # Finalize per-operator dictionary with metrics
    final_by_op: Dict[str, Dict[str, Any]] = {}
    for op, stats in by_operator_counts.items():
        corr = stats["correct"]
        tot = stats["total"]
        errs = stats["errors"]
        acc = corr / tot if tot > 0 else 0.0
        final_by_op[op] = {
            "symbol": op,
            "name": OPERATOR_NAMES.get(op, op),
            "correct": corr,
            "total": tot,
            "errors": errs,
            "accuracy": acc,
            "error_rate": 1.0 - acc,
        }

    return ExactMatchResult(
        accuracy=accuracy,
        correct=total_correct,
        total=total_samples,
        by_operator=final_by_op,
        samples=all_samples,
        errors_by_operator=errors_by_operator,
        samples_by_operator=samples_by_operator,
        split_name=split_name,
        operator_filter=filter_symbols,
    )


def exact_match_by_operator(
    model: Optional[Union[GPT, str, Path]] = None,
    data: Union[str, Path, torch.Tensor, List[str]] = "test",
    checkpoint_path: Union[str, Path] = "artifacts/model.pt",
    batch_size: int = 512,
    device: Optional[Union[str, torch.device]] = None,
    return_details: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience helper that evaluates and returns a clean dictionary mapping
    each operator symbol to its accuracy and error statistics.

    Example return:
    {
        '+': {'name': 'Addition', 'accuracy': 1.0, 'correct': 1000, 'total': 1000, 'errors': 0},
        ...
    }
    """
    res = exact_match(
        model=model,
        data=data,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        device=device,
        return_details=return_details,
    )
    if isinstance(res, dict) and "test" in res:
        # If 'both' was passed
        return {split: r.by_operator for split, r in res.items()}
    return res.by_operator


def operator_accuracy(
    op: str,
    model: Optional[Union[GPT, str, Path]] = None,
    data: Union[str, Path, torch.Tensor, List[str]] = "test",
    checkpoint_path: Union[str, Path] = "artifacts/model.pt",
    batch_size: int = 512,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    Evaluates and returns the exact match accuracy for a single operator.
    Example: operator_accuracy('+', model, 'test') -> 1.0
    """
    res = exact_match(
        model=model,
        data=data,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        device=device,
        operator=op,
    )
    normalized = normalize_operator(op)
    return res.op_accuracy(normalized)


def evaluate_saved_model(
    checkpoint_path: Union[str, Path] = "artifacts/model.pt",
    batch_size: int = 512,
    device: Optional[str] = None,
    operator: Optional[Union[str, List[str]]] = None,
    show_table: bool = True,
    show_errors: bool = True,
) -> Dict[str, ExactMatchResult]:
    """
    Evaluates the saved model on both train and test sets and prints a detailed report
    including per-operator table and error analysis.
    """
    print(f"Loading checkpoint from '{checkpoint_path}'...")
    model, cfg = load_saved_model(checkpoint_path, device=device)
    print(f"Model loaded successfully on device '{model.config.device}'.\n")

    print("Evaluating Exact Match on Train set...")
    train_res = exact_match(
        model, data="train", batch_size=batch_size, device=device, operator=operator
    )
    print(train_res.summary(show_table=show_table, show_errors=False))
    print()

    print("Evaluating Exact Match on Test set...")
    test_res = exact_match(
        model, data="test", batch_size=batch_size, device=device, operator=operator
    )
    print(test_res.summary(show_table=show_table, show_errors=show_errors))
    print()

    return {"train": train_res, "test": test_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate exact match accuracy of saved nanoCalc-GPT model.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/model.pt", help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    parser.add_argument("--operator", "-op", type=str, default=None, help="Filter by operator (+, -, *, /, or name like 'multiplication')")
    parser.add_argument("--show-table", action="store_true", default=True, help="Display formatted ASCII table")
    parser.add_argument("--show-errors", action="store_true", default=False, help="Display error samples per operator")
    args = parser.parse_args()

    if args.split == "both":
        evaluate_saved_model(
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            operator=args.operator,
            show_table=args.show_table,
            show_errors=args.show_errors,
        )
    else:
        model, _ = load_saved_model(args.checkpoint, device=args.device)
        result = exact_match(
            model,
            data=args.split,
            batch_size=args.batch_size,
            device=args.device,
            operator=args.operator,
        )
        print(result.summary(show_table=args.show_table, show_errors=args.show_errors))

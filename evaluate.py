"""
Script to evaluate saved nanoCalc-GPT model on train and test datasets.
Usage:
    python evaluate.py
    python evaluate.py --split test
    python evaluate.py --split test --operator "*"
    python evaluate.py --split test --show-errors
"""
import argparse
from exact_match import (
    evaluate_saved_model,
    exact_match,
    exact_match_by_operator,
    load_saved_model,
    operator_accuracy,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved nanoCalc-GPT model accuracy.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/model.pt", help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu, cuda)")
    parser.add_argument("--operator", "-op", type=str, default=None, help="Operator symbol or alias (+, -, *, /, add, mul, etc.)")
    parser.add_argument("--show-errors", action="store_true", default=False, help="Display error samples per operator")
    args = parser.parse_args()

    if args.split == "both":
        evaluate_saved_model(
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            operator=args.operator,
            show_table=True,
            show_errors=args.show_errors,
        )
    else:
        model, _ = load_saved_model(args.checkpoint, device=args.device)
        res = exact_match(
            model,
            data=args.split,
            batch_size=args.batch_size,
            device=args.device,
            operator=args.operator,
        )
        print(res.summary(show_table=True, show_errors=args.show_errors))

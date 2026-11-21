"""Command-line feature extraction, training, and held-out evaluation."""

import argparse
import json
from pathlib import Path

from .config import BENCHMARKS
from .data import extract
from .features import MANTIS_REPO
from .training import evaluate, train


def main():
    parser = argparse.ArgumentParser(prog="diremoe", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract_parser = commands.add_parser("extract", help="Encode preprocessed signals into a feature store.")
    extract_parser.add_argument("--input", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--dataset", choices=BENCHMARKS, required=True)
    extract_parser.add_argument("--weights", default=MANTIS_REPO, help="MantisV2 repository or local directory.")
    extract_parser.add_argument('--regional', action='store_true', help='Store per-node temporal embeddings.')
    extract_parser.add_argument("--physical-lags", type=float, nargs=2, metavar=("SHORT_S", "LONG_S"),
                                help="Two positive delays in seconds. Input NPZ must include tr_seconds.")
    extract_parser.add_argument("--batch-size", type=int, default=8)

    train_parser = commands.add_parser("train", help="Fit a fold and save the validation-selected checkpoint.")
    train_parser.add_argument('--temporal-groups', type=Path, help='NPY vector of zero-based node group IDs.')
    train_parser.add_argument('--direction-groups', type=Path, help='NPY vector of zero-based node group IDs.')
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser = commands.add_parser("evaluate", help="Evaluate a saved checkpoint on held-out indices.")
    for command in (train_parser, evaluate_parser):
        command.add_argument("--features", type=Path, required=True)
        command.add_argument("--split", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
    for command in (extract_parser, train_parser, evaluate_parser):
        command.add_argument("--device", default="cpu", help="PyTorch device, for example cuda:0.")

    args = parser.parse_args()
    if args.command == "extract":
        if args.batch_size < 1:
            extract_parser.error("--batch-size must be positive")
        extract(args.input, args.output, args.dataset, args.weights, args.device, args.batch_size, args.physical_lags, args.regional)
        return
    if args.command == "train":
        if args.epochs < 1 or not 0 <= args.seed < 2**32:
            train_parser.error("--epochs must be positive and --seed must be in [0, 2**32)")
        result = train(args.features, args.split, args.checkpoint, args.epochs, args.seed, args.device,
                       temporal_groups=args.temporal_groups, direction_groups=args.direction_groups)
    else:
        result = evaluate(args.features, args.split, args.checkpoint, args.device)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Backward-compatible entrypoint. Prefer src.training.train_machine_learning."""

from src.training.train_machine_learning import parse_args, train


if __name__ == "__main__":
    args = parse_args()
    train(args.data_csv, args.output_dir, args.test_size, args.seed)

import argparse
import sys

from ai_pipeline.cli.commands import (
    build_services,
    command_dataset_add,
    command_dataset_list,
    command_evaluate,
    command_pipeline,
    command_preprocess,
    command_train,
    command_validate,
)
from ai_pipeline.logic.exceptions import PipelineError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-pipeline",
        description="AI Data & Training Manager",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Manage datasets",
    )

    dataset_subparsers = (
        dataset_parser.add_subparsers(
            dest="dataset_command",
            required=True,
        )
    )

    add_parser = dataset_subparsers.add_parser(
        "add",
        help="Register a dataset",
    )
    add_parser.add_argument("path")
    add_parser.add_argument(
        "--name",
        required=True,
    )

    dataset_subparsers.add_parser(
        "list",
        help="List registered datasets",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a CSV dataset",
    )
    validate_parser.add_argument("path")
    validate_parser.add_argument(
        "--label",
        default="label",
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Clean and preprocess a dataset",
    )
    preprocess_parser.add_argument("input")
    preprocess_parser.add_argument("output")
    preprocess_parser.add_argument(
        "--label",
        default="label",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a KNN model",
    )
    train_parser.add_argument("dataset")
    train_parser.add_argument("model")
    train_parser.add_argument(
        "--k",
        type=int,
        default=3,
    )
    train_parser.add_argument(
        "--label",
        default="label",
    )
    train_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved model",
    )
    evaluate_parser.add_argument("dataset")
    evaluate_parser.add_argument("model")
    evaluate_parser.add_argument(
        "--results",
        default="results",
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run pipeline operations",
    )

    pipeline_subparsers = (
        pipeline_parser.add_subparsers(
            dest="pipeline_command",
            required=True,
        )
    )

    run_parser = pipeline_subparsers.add_parser(
        "run",
        help="Run validate -> preprocess -> train -> evaluate",
    )

    run_parser.add_argument("input")
    run_parser.add_argument(
        "--output",
        default="data/processed/classification_clean.csv",
    )
    run_parser.add_argument(
        "--model",
        default="models/knn_model.json",
    )
    run_parser.add_argument(
        "--results",
        default="results",
    )
    run_parser.add_argument(
        "--label",
        default="label",
    )
    run_parser.add_argument(
        "--k",
        type=int,
        default=3,
    )
    run_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )

    return parser


def main() -> int:
    if len(sys.argv) == 1:
        print("AI Data & Training Manager")
        print("Use --help to view available commands.")
        return 0

    parser = build_parser()
    args = parser.parse_args()
    services = build_services()

    try:
        if args.command == "dataset":
            if args.dataset_command == "add":
                command_dataset_add(
                    args,
                    services,
                )
            else:
                command_dataset_list(
                    args,
                    services,
                )

        elif args.command == "validate":
            command_validate(
                args,
                services,
            )

        elif args.command == "preprocess":
            command_preprocess(
                args,
                services,
            )

        elif args.command == "train":
            command_train(
                args,
                services,
            )

        elif args.command == "evaluate":
            command_evaluate(
                args,
                services,
            )

        elif args.command == "pipeline":
            command_pipeline(
                args,
                services,
            )

        return 0

    except (
        PipelineError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

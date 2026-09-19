from ai_pipeline.io.csv_repository import CsvRepository
from ai_pipeline.io.json_repository import JsonRepository
from ai_pipeline.io.model_repository import ModelRepository
from ai_pipeline.logic.services.dataset_service import DatasetService
from ai_pipeline.logic.services.evaluation_service import (
    EvaluationService,
)
from ai_pipeline.logic.services.preprocessing_service import (
    PreprocessingService,
)
from ai_pipeline.logic.services.training_service import (
    TrainingService,
)
from ai_pipeline.logic.validators import (
    validate_csv_path,
    validate_records,
)


def build_services() -> dict:
    csv_repository = CsvRepository()
    json_repository = JsonRepository()
    model_repository = ModelRepository(
        json_repository
    )

    return {
        "csv": csv_repository,
        "json": json_repository,
        "dataset": DatasetService(
            csv_repository,
            json_repository,
        ),
        "preprocess": PreprocessingService(
            csv_repository
        ),
        "training": TrainingService(
            csv_repository,
            model_repository,
        ),
        "evaluation": EvaluationService(
            csv_repository,
            model_repository,
            json_repository,
        ),
    }


def command_dataset_add(
    args,
    services,
) -> None:
    dataset = services["dataset"].register(
        args.path,
        args.name,
    )

    print("Dataset registered successfully.")
    print(f"ID: {dataset.dataset_id}")
    print(f"Name: {dataset.name}")
    print(f"Rows: {dataset.rows}")
    print(
        "Columns: "
        + ", ".join(dataset.columns)
    )


def command_dataset_list(
    args,
    services,
) -> None:
    datasets = services["dataset"].list_all()

    if not datasets:
        print("No datasets registered.")
        return

    for dataset in datasets:
        print(
            f"{dataset.dataset_id} | "
            f"{dataset.name} | "
            f"{dataset.rows} rows | "
            f"{dataset.path}"
        )


def command_validate(
    args,
    services,
) -> None:
    path = validate_csv_path(args.path)

    records = services["csv"].read(path)

    validate_records(
        records,
        args.label,
    )

    print("Validation successful.")
    print(f"Rows: {len(records)}")
    print(
        "Columns: "
        + ", ".join(records[0])
    )


def command_preprocess(
    args,
    services,
) -> None:
    stats = services["preprocess"].process(
        args.input,
        args.output,
        args.label,
    )

    print("Preprocessing completed.")
    print(f"Input rows: {stats['input_rows']}")
    print(f"Output rows: {stats['output_rows']}")
    print(f"Removed rows: {stats['removed_rows']}")
    print(f"Output: {args.output}")


def command_train(
    args,
    services,
) -> None:
    model = services["training"].train(
        args.dataset,
        args.model,
        args.k,
        args.label,
        args.train_ratio,
    )

    print("Training completed.")
    print(f"Algorithm: {model.model_type}")
    print(f"k: {model.k}")
    print(
        "Features: "
        + ", ".join(model.feature_columns)
    )
    print(
        f"Training rows: {len(model.train_rows)}"
    )
    print(f"Model: {args.model}")


def command_evaluate(
    args,
    services,
) -> None:
    experiment = services["evaluation"].evaluate(
        args.dataset,
        args.model,
        args.results,
    )

    print("Evaluation completed.")
    print(
        f"Experiment: "
        f"{experiment.experiment_id}"
    )
    print(
        f"Accuracy: "
        f"{experiment.accuracy:.2%}"
    )
    print(
        f"Train rows: "
        f"{experiment.train_rows}"
    )
    print(
        f"Test rows: "
        f"{experiment.test_rows}"
    )

def command_pipeline(
    args,
    services,
) -> None:
    print("=== AI DATA PIPELINE ===")

    input_path = validate_csv_path(args.input)
    records = services["csv"].read(input_path)

    validate_records(
        records,
        args.label,
    )

    print("Validation successful.")
    print(f"Rows: {len(records)}")
    print(
        "Columns: "
        + ", ".join(records[0])
    )

    preprocessing_stats = services["preprocess"].process(
        args.input,
        args.output,
        args.label,
    )

    print("Preprocessing completed.")
    print(
        f"Input rows: "
        f"{preprocessing_stats['input_rows']}"
    )
    print(
        f"Output rows: "
        f"{preprocessing_stats['output_rows']}"
    )
    print(
        f"Removed rows: "
        f"{preprocessing_stats['removed_rows']}"
    )
    print(f"Output: {args.output}")

    model = services["training"].train(
        args.output,
        args.model,
        args.k,
        args.label,
        args.train_ratio,
    )

    print("Training completed.")
    print(f"Algorithm: {model.model_type}")
    print(f"k: {model.k}")
    print(
        "Features: "
        + ", ".join(model.feature_columns)
    )
    print(
        f"Training rows: "
        f"{len(model.train_rows)}"
    )
    print(f"Model: {args.model}")

    experiment = services["evaluation"].evaluate(
        args.output,
        args.model,
        args.results,
    )

    print("Evaluation completed.")
    print(
        f"Experiment: "
        f"{experiment.experiment_id}"
    )
    print(
        f"Accuracy: "
        f"{experiment.accuracy:.2%}"
    )
    print(
        f"Train rows: "
        f"{experiment.train_rows}"
    )
    print(
        f"Test rows: "
        f"{experiment.test_rows}"
    )

    print("=== PIPELINE FINISHED ===")

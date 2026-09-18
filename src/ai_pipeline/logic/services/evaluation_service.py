from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ai_pipeline.io.csv_repository import CsvRepository
from ai_pipeline.io.json_repository import JsonRepository
from ai_pipeline.io.model_repository import ModelRepository
from ai_pipeline.logic.models.experiment import Experiment
from ai_pipeline.logic.services.training_service import TrainingService
from ai_pipeline.logic.validators import (
    validate_csv_path,
    validate_records,
)


class EvaluationService:
    def __init__(
        self,
        csv_repository: CsvRepository,
        model_repository: ModelRepository,
        json_repository: JsonRepository,
    ) -> None:
        self._csv = csv_repository
        self._models = model_repository
        self._json = json_repository

    def evaluate(
        self,
        dataset_path: str,
        model_path: str,
        results_directory: str = "results",
    ) -> Experiment:
        dataset = validate_csv_path(dataset_path)

        model = self._models.load(model_path)

        records = self._csv.read(dataset)

        validate_records(
            records,
            model.label_column,
        )

        _, test_rows = TrainingService.split_records(
            records,
            model.train_ratio,
        )

        correct = 0
        predictions: list[dict[str, str]] = []

        for row in test_rows:
            prediction = TrainingService.predict(
                model,
                row,
            )

            actual = row[model.label_column]

            predictions.append(
                {
                    "actual": actual,
                    "predicted": prediction,
                }
            )

            if prediction == actual:
                correct += 1

        accuracy = (
            correct / len(test_rows)
            if test_rows
            else 0.0
        )

        experiment = Experiment(
            experiment_id=str(uuid4())[:8],
            dataset_path=str(dataset),
            model_path=str(model_path),
            accuracy=accuracy,
            train_rows=len(records) - len(test_rows),
            test_rows=len(test_rows),
            created_at=datetime.now().isoformat(
                timespec="seconds"
            ),
        )

        result_path = (
            Path(results_directory)
            / f"experiment_{experiment.experiment_id}.json"
        )

        self._json.write(
            result_path,
            {
                "experiment": experiment.to_dict(),
                "predictions": predictions,
            },
        )

        return experiment

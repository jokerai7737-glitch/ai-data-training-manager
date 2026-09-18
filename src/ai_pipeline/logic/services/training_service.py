from datetime import datetime
import math

from ai_pipeline.io.csv_repository import CsvRepository
from ai_pipeline.io.model_repository import ModelRepository
from ai_pipeline.logic.exceptions import TrainingError
from ai_pipeline.logic.models.model_artifact import ModelArtifact
from ai_pipeline.logic.validators import (
    validate_csv_path,
    validate_k,
    validate_records,
    validate_train_ratio,
)


class TrainingService:
    """Train and use a small KNN classification model."""

    def __init__(
        self,
        csv_repository: CsvRepository,
        model_repository: ModelRepository,
    ) -> None:
        self._csv = csv_repository
        self._models = model_repository

    @staticmethod
    def split_records(
        records: list[dict[str, str]],
        train_ratio: float,
    ) -> tuple[
        list[dict[str, str]],
        list[dict[str, str]],
    ]:
        cut = int(len(records) * train_ratio)
        cut = max(1, min(cut, len(records) - 1))

        return records[:cut], records[cut:]

    @staticmethod
    def distance(
        first: dict[str, str],
        second: dict[str, str],
        feature_columns: list[str],
    ) -> float:
        return math.sqrt(
            sum(
                (
                    float(first[column])
                    - float(second[column])
                ) ** 2
                for column in feature_columns
            )
        )

    @classmethod
    def predict(
        cls,
        model: ModelArtifact,
        record: dict[str, str],
    ) -> str:
        scored = [
            (
                cls.distance(
                    record,
                    train_row,
                    model.feature_columns,
                ),
                train_row[model.label_column],
            )
            for train_row in model.train_rows
        ]

        nearest = sorted(
            scored,
            key=lambda item: item[0],
        )[: model.k]

        votes: dict[str, int] = {}

        for _, label in nearest:
            votes[label] = votes.get(label, 0) + 1

        return max(
            votes,
            key=votes.get,
        )

    def train(
        self,
        dataset_path: str,
        model_path: str,
        k: int = 3,
        label_column: str = "label",
        train_ratio: float = 0.8,
    ) -> ModelArtifact:
        dataset = validate_csv_path(dataset_path)

        k = validate_k(k)
        train_ratio = validate_train_ratio(
            train_ratio
        )

        records = self._csv.read(dataset)

        if len(records) < 3:
            raise TrainingError(
                "At least 3 records are required."
            )

        validate_records(
            records,
            label_column,
        )

        feature_columns = [
            column
            for column in records[0]
            if column != label_column
        ]

        if not feature_columns:
            raise TrainingError(
                "Dataset must contain at least one feature."
            )

        for row in records:
            for column in feature_columns:
                try:
                    float(row[column])
                except (TypeError, ValueError) as exc:
                    raise TrainingError(
                        f"Feature '{column}' must be numeric."
                    ) from exc

        train_rows, _ = self.split_records(
            records,
            train_ratio,
        )

        model = ModelArtifact(
            model_type="KNN",
            k=k,
            label_column=label_column,
            feature_columns=feature_columns,
            train_rows=train_rows,
            train_ratio=train_ratio,
            created_at=datetime.now().isoformat(
                timespec="seconds"
            ),
        )

        self._models.save(
            model_path,
            model,
        )

        return model

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ai_pipeline.io.csv_repository import CsvRepository
from ai_pipeline.io.json_repository import JsonRepository
from ai_pipeline.logic.models.dataset import Dataset
from ai_pipeline.logic.validators import (
    validate_csv_path,
    validate_non_empty,
)


class DatasetService:
    def __init__(
        self,
        csv_repository: CsvRepository,
        json_repository: JsonRepository,
    ) -> None:
        self._csv = csv_repository
        self._json = json_repository
        self._registry = Path("data/datasets.json")

    def register(
        self,
        path: str,
        name: str,
    ) -> Dataset:
        dataset_path = validate_csv_path(path)
        dataset_name = validate_non_empty(name, "Dataset name")

        records = self._csv.read(dataset_path)
        columns = list(records[0]) if records else []

        dataset = Dataset(
            dataset_id=str(uuid4())[:8],
            name=dataset_name,
            path=str(dataset_path),
            rows=len(records),
            columns=columns,
            created_at=datetime.now().isoformat(
                timespec="seconds"
            ),
        )

        datasets = self._json.read(
            self._registry,
            default=[],
        )

        datasets.append(dataset.to_dict())

        self._json.write(
            self._registry,
            datasets,
        )

        return dataset

    def list_all(self) -> list[Dataset]:
        data = self._json.read(
            self._registry,
            default=[],
        )

        return [
            Dataset.from_dict(item)
            for item in data
        ]

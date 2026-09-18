from ai_pipeline.io.csv_repository import CsvRepository
from ai_pipeline.logic.validators import (
    validate_csv_path,
    validate_records,
)


class PreprocessingService:
    def __init__(
        self,
        csv_repository: CsvRepository,
    ) -> None:
        self._csv = csv_repository

    @staticmethod
    def clean_records(
        records: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []

        for record in records:
            normalized = {
                key.strip(): (
                    value.strip()
                    if value is not None
                    else ""
                )
                for key, value in record.items()
            }

            if all(normalized.values()):
                cleaned.append(normalized)

        return cleaned

    def process(
        self,
        input_path: str,
        output_path: str,
        label_column: str = "label",
    ) -> dict[str, int]:
        source = validate_csv_path(input_path)
        records = self._csv.read(source)

        cleaned = self.clean_records(records)

        validate_records(
            cleaned,
            label_column,
        )

        self._csv.write(
            output_path,
            cleaned,
            list(cleaned[0]),
        )

        return {
            "input_rows": len(records),
            "output_rows": len(cleaned),
            "removed_rows": (
                len(records) - len(cleaned)
            ),
        }

from pathlib import Path

from ai_pipeline.logic.exceptions import ValidationError


def validate_csv_path(path: str | Path) -> Path:
    file_path = Path(path)

    if not file_path.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValidationError("Input file must have a .csv extension.")

    return file_path


def validate_non_empty(value: str, field_name: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValidationError(f"{field_name} cannot be empty.")

    return cleaned


def validate_k(k: int) -> int:
    if k <= 0:
        raise ValidationError("k must be greater than zero.")

    return k


def validate_train_ratio(ratio: float) -> float:
    if not 0.5 <= ratio < 1.0:
        raise ValidationError(
            "train_ratio must be between 0.5 and 0.99."
        )

    return ratio


def validate_records(
    records: list[dict[str, str]],
    label_column: str = "label",
) -> None:
    if not records:
        raise ValidationError("Dataset contains no records.")

    columns = list(records[0])

    if label_column not in columns:
        raise ValidationError(
            f"Required label column '{label_column}' was not found."
        )

    for row_number, record in enumerate(records, start=2):
        if set(record) != set(columns):
            raise ValidationError(
                f"Row {row_number} has inconsistent columns."
            )

        if any(
            value is None or str(value).strip() == ""
            for value in record.values()
        ):
            raise ValidationError(
                f"Row {row_number} contains an empty value."
            )

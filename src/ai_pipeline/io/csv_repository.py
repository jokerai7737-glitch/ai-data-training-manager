import csv
from pathlib import Path

from ai_pipeline.logic.exceptions import RepositoryError


class CsvRepository:
    """CSV file operations only."""

    def read(self, path: str | Path) -> list[dict[str, str]]:
        file_path = Path(path)

        try:
            with file_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                return list(csv.DictReader(file))
        except OSError as exc:
            raise RepositoryError(
                f"Could not read CSV file: {file_path}"
            ) from exc

    def write(
        self,
        path: str | Path,
        rows: list[dict[str, str]],
        fieldnames: list[str],
    ) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with file_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            raise RepositoryError(
                f"Could not write CSV file: {file_path}"
            ) from exc

import json
from pathlib import Path
from typing import Any

from ai_pipeline.logic.exceptions import RepositoryError


class JsonRepository:
    """JSON file operations only."""

    def read(
        self,
        path: str | Path,
        default: Any = None,
    ) -> Any:
        file_path = Path(path)

        if not file_path.exists():
            return default

        try:
            with file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                f"Could not read JSON file: {file_path}"
            ) from exc

    def write(
        self,
        path: str | Path,
        data: Any,
    ) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with file_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    data,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
        except OSError as exc:
            raise RepositoryError(
                f"Could not write JSON file: {file_path}"
            ) from exc

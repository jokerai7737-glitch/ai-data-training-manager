from pathlib import Path

from ai_pipeline.io.json_repository import JsonRepository
from ai_pipeline.logic.exceptions import RepositoryError
from ai_pipeline.logic.models.model_artifact import ModelArtifact


class ModelRepository:
    """Persistence for trained model artifacts."""

    def __init__(
        self,
        repository: JsonRepository | None = None,
    ) -> None:
        self._repository = repository or JsonRepository()

    def save(
        self,
        path: str | Path,
        model: ModelArtifact,
    ) -> None:
        self._repository.write(path, model.to_dict())

    def load(self, path: str | Path) -> ModelArtifact:
        try:
            data = self._repository.read(path)
        except RepositoryError:
            raise

        if not data:
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        return ModelArtifact.from_dict(data)

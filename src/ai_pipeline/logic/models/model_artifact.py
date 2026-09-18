from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ModelArtifact:
    model_type: str
    k: int
    label_column: str
    feature_columns: list[str]
    train_rows: list[dict[str, str]]
    train_ratio: float
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelArtifact":
        return cls(**data)

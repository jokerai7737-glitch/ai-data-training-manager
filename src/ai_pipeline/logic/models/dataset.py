from dataclasses import asdict, dataclass


@dataclass(slots=True)
class Dataset:
    dataset_id: str
    name: str
    path: str
    rows: int
    columns: list[str]
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        return cls(**data)

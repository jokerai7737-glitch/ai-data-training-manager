from dataclasses import asdict, dataclass


@dataclass(slots=True)
class Experiment:
    experiment_id: str
    dataset_path: str
    model_path: str
    accuracy: float
    train_rows: int
    test_rows: int
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

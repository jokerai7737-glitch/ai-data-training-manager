from pathlib import Path
import json

import pytest

from ai_pipeline.cli.main import build_parser
from ai_pipeline.io.csv_repository import CsvRepository
from ai_pipeline.io.json_repository import JsonRepository
from ai_pipeline.io.model_repository import ModelRepository
from ai_pipeline.logic.exceptions import TrainingError, ValidationError
from ai_pipeline.logic.models.dataset import Dataset
from ai_pipeline.logic.models.experiment import Experiment
from ai_pipeline.logic.models.model_artifact import ModelArtifact
from ai_pipeline.logic.services.evaluation_service import EvaluationService
from ai_pipeline.logic.services.preprocessing_service import PreprocessingService
from ai_pipeline.logic.services.training_service import TrainingService
from ai_pipeline.logic.validators import (
    validate_csv_path,
    validate_k,
    validate_non_empty,
    validate_records,
    validate_train_ratio,
)


def sample_rows():
    return [
        {"age": "20", "income": "20000", "label": "junior"},
        {"age": "30", "income": "40000", "label": "mid"},
        {"age": "40", "income": "60000", "label": "senior"},
    ]


def write_sample_csv(path: Path):
    path.write_text(
        "age,income,label\n"
        "20,20000,junior\n"
        "22,22000,junior\n"
        "30,40000,mid\n"
        "32,42000,mid\n"
        "40,60000,senior\n"
        "42,62000,senior\n",
        encoding="utf-8",
    )


def test_validate_csv_path_accepts_existing_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert validate_csv_path(path) == path


def test_validate_csv_path_rejects_missing_file(tmp_path):
    with pytest.raises(ValidationError):
        validate_csv_path(tmp_path / "missing.csv")


def test_validate_csv_path_rejects_wrong_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValidationError):
        validate_csv_path(path)


def test_validate_non_empty():
    assert validate_non_empty("  khaled  ", "name") == "khaled"


def test_validate_non_empty_rejects_empty():
    with pytest.raises(ValidationError):
        validate_non_empty("   ", "name")


def test_validate_k_accepts_positive_value():
    assert validate_k(3) == 3


def test_validate_k_rejects_zero():
    with pytest.raises(ValidationError):
        validate_k(0)


def test_validate_train_ratio_accepts_valid_ratio():
    assert validate_train_ratio(0.8) == 0.8


def test_validate_train_ratio_rejects_invalid_ratio():
    with pytest.raises(ValidationError):
        validate_train_ratio(0.2)


def test_validate_records_accepts_valid_records():
    validate_records(sample_rows(), "label")


def test_validate_records_rejects_empty_records():
    with pytest.raises(ValidationError):
        validate_records([], "label")


def test_validate_records_rejects_missing_label():
    with pytest.raises(ValidationError):
        validate_records([{"age": "20"}], "label")


def test_validate_records_rejects_inconsistent_columns():
    rows = [
        {"age": "20", "label": "junior"},
        {"age": "30", "income": "40000", "label": "mid"},
    ]
    with pytest.raises(ValidationError):
        validate_records(rows, "label")


def test_validate_records_rejects_empty_value():
    rows = [{"age": "", "income": "20000", "label": "junior"}]
    with pytest.raises(ValidationError):
        validate_records(rows, "label")


def test_dataset_model_round_trip():
    dataset = Dataset("1", "sample", "data.csv", 3, ["age", "label"], "now")
    restored = Dataset.from_dict(dataset.to_dict())
    assert restored == dataset


def test_model_artifact_round_trip():
    model = ModelArtifact(
        "KNN", 3, "label", ["age"], sample_rows(), 0.8, "now"
    )
    restored = ModelArtifact.from_dict(model.to_dict())
    assert restored == model


def test_experiment_to_dict():
    experiment = Experiment("1", "data.csv", "model.json", 0.9, 8, 2, "now")
    data = experiment.to_dict()
    assert data["accuracy"] == 0.9
    assert data["test_rows"] == 2


def test_csv_repository_write_and_read(tmp_path):
    repository = CsvRepository()
    path = tmp_path / "data.csv"
    rows = sample_rows()
    repository.write(path, rows, ["age", "income", "label"])
    assert repository.read(path) == rows


def test_json_repository_write_and_read(tmp_path):
    repository = JsonRepository()
    path = tmp_path / "data.json"
    data = {"name": "sample", "rows": 3}
    repository.write(path, data)
    assert repository.read(path) == data


def test_json_repository_returns_default_for_missing_file(tmp_path):
    repository = JsonRepository()
    assert repository.read(tmp_path / "missing.json", default=[]) == []


def test_model_repository_save_and_load(tmp_path):
    repository = ModelRepository()
    path = tmp_path / "model.json"
    model = ModelArtifact(
        "KNN", 3, "label", ["age"], sample_rows(), 0.8, "now"
    )
    repository.save(path, model)
    assert repository.load(path) == model


def test_preprocessing_clean_records_strips_values():
    records = [{" age ": " 20 ", " label ": " junior "}]
    cleaned = PreprocessingService.clean_records(records)
    assert cleaned == [{"age": "20", "label": "junior"}]


def test_preprocessing_clean_records_removes_empty_rows():
    records = [
        {"age": "20", "label": "junior"},
        {"age": "", "label": "mid"},
    ]
    cleaned = PreprocessingService.clean_records(records)
    assert cleaned == [{"age": "20", "label": "junior"}]


def test_preprocessing_process_creates_output(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    write_sample_csv(input_path)
    service = PreprocessingService(CsvRepository())
    stats = service.process(input_path, output_path)
    assert output_path.exists()
    assert stats["input_rows"] == 6
    assert stats["output_rows"] == 6


def test_training_split_records():
    train, test = TrainingService.split_records(sample_rows(), 0.66)
    assert len(train) == 1
    assert len(test) == 2


def test_training_distance():
    distance = TrainingService.distance(
        {"age": "20", "income": "20000"},
        {"age": "23", "income": "24000"},
        ["age", "income"],
    )
    assert distance > 0


def test_training_predict():
    model = ModelArtifact(
        "KNN",
        3,
        "label",
        ["age", "income"],
        [
            {"age": "20", "income": "20000", "label": "junior"},
            {"age": "21", "income": "21000", "label": "junior"},
            {"age": "40", "income": "60000", "label": "senior"},
        ],
        0.8,
        "now",
    )
    prediction = TrainingService.predict(
        model,
        {"age": "22", "income": "22000", "label": "junior"},
    )
    assert prediction == "junior"


def test_training_creates_model(tmp_path):
    input_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.json"
    write_sample_csv(input_path)
    service = TrainingService(CsvRepository(), ModelRepository())
    model = service.train(input_path, model_path)
    assert model_path.exists()
    assert model.model_type == "KNN"
    assert model.k == 3


def test_training_rejects_non_numeric_features(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "age,income,label\n"
        "twenty,20000,junior\n"
        "30,40000,mid\n"
        "40,60000,senior\n",
        encoding="utf-8",
    )
    service = TrainingService(CsvRepository(), ModelRepository())
    with pytest.raises(TrainingError):
        service.train(path, tmp_path / "model.json")


def test_evaluation_creates_result(tmp_path):
    dataset_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.json"
    results_path = tmp_path / "results"
    write_sample_csv(dataset_path)

    model_repository = ModelRepository()
    model = TrainingService(CsvRepository(), model_repository).train(
        dataset_path, model_path
    )

    service = EvaluationService(
        CsvRepository(),
        model_repository,
        JsonRepository(),
    )

    experiment = service.evaluate(
        dataset_path,
        model_path,
        results_path,
    )

    assert 0.0 <= experiment.accuracy <= 1.0
    assert list(results_path.glob("experiment_*.json"))


def test_cli_parser_validate_command():
    parser = build_parser()
    args = parser.parse_args(["validate", "data.csv"])
    assert args.command == "validate"
    assert args.label == "label"


def test_cli_parser_pipeline_command():
    parser = build_parser()
    args = parser.parse_args(["pipeline", "run", "data.csv"])
    assert args.command == "pipeline"
    assert args.pipeline_command == "run"
    assert args.k == 3

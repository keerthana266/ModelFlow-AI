# app/services/storage.py

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parents[2]
STORAGE_DIR = BASE_DIR / "storage"

DATASET_DIR = STORAGE_DIR / "datasets"
MODEL_DIR = STORAGE_DIR / "models"
EXPERIMENT_DIR = STORAGE_DIR / "experiments"
DRIFT_REPORT_DIR = STORAGE_DIR / "drift_reports"
DRIFT_REFERENCE_DIR = STORAGE_DIR / "drift_reference"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_storage_folders() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def save_uploaded_dataset(file, max_upload_mb: int = 50) -> dict:
    create_storage_folders()

    original_filename = file.filename or "uploaded.csv"
    file_extension = Path(original_filename).suffix.lower()

    if file_extension != ".csv":
        raise ValueError("Only CSV files are allowed.")

    dataset_id = generate_id("dataset")
    saved_path = DATASET_DIR / f"{dataset_id}.csv"

    max_bytes = max_upload_mb * 1024 * 1024
    bytes_written = 0

    try:
        with saved_path.open("wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                bytes_written += len(chunk)

                if bytes_written > max_bytes:
                    buffer.close()
                    saved_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"File is too large. Max allowed size is {max_upload_mb} MB."
                    )

                buffer.write(chunk)

    finally:
        file.file.close()

    return {
        "dataset_id": dataset_id,
        "original_filename": original_filename,
        "saved_path": str(saved_path),
        "uploaded_at": utc_now_iso(),
    }


def get_dataset_path(dataset_id: str) -> Path:
    dataset_path = DATASET_DIR / f"{dataset_id}.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' was not found.")

    return dataset_path


def delete_dataset_file(dataset_id: str) -> None:
    dataset_path = DATASET_DIR / f"{dataset_id}.csv"
    dataset_path.unlink(missing_ok=True)


def get_model_path(model_id: str) -> Path:
    model_path = MODEL_DIR / f"{model_id}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model '{model_id}' was not found.")

    return model_path


def save_experiment_result(experiment_id: str, result: dict) -> Path:
    create_storage_folders()

    experiment_path = EXPERIMENT_DIR / f"{experiment_id}.json"

    with experiment_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, default=str)

    return experiment_path


def load_experiment_result(experiment_id: str) -> dict:
    experiment_path = EXPERIMENT_DIR / f"{experiment_id}.json"

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment '{experiment_id}' was not found.")

    with experiment_path.open("r", encoding="utf-8") as f:
        return json.load(f)
# app/ml/drift.py

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from app.services.storage import (
    DRIFT_REFERENCE_DIR,
    DRIFT_REPORT_DIR,
    get_dataset_path,
    generate_id,
)


def save_reference_data(model_id: str, X: pd.DataFrame, max_rows: int = 5000) -> Path:
    """
    Saves the training feature data as reference data for future drift checks.
    """

    DRIFT_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    reference_df = X.copy()

    if len(reference_df) > max_rows:
        reference_df = reference_df.sample(max_rows, random_state=42)

    path = DRIFT_REFERENCE_DIR / f"{model_id}.csv"
    reference_df.to_csv(path, index=False)

    return path


def load_reference_data(model_id: str) -> pd.DataFrame:
    reference_path = DRIFT_REFERENCE_DIR / f"{model_id}.csv"

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference data was not found for model '{model_id}'. "
            "Train this model again after enabling drift monitoring."
        )

    return pd.read_csv(reference_path)


def build_current_dataframe(
    current_dataset_id: Optional[str],
    current_data: Optional[List[Dict[str, Any]]],
) -> pd.DataFrame:
    if current_dataset_id:
        dataset_path = get_dataset_path(current_dataset_id)
        return pd.read_csv(dataset_path)

    if current_data:
        return pd.DataFrame(current_data)

    raise ValueError("Provide either current_dataset_id or current_data.")


def align_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_aligned = reference_df.copy()
    current_aligned = current_df.copy()

    for column in feature_columns:
        if column not in reference_aligned.columns:
            reference_aligned[column] = None

        if column not in current_aligned.columns:
            current_aligned[column] = None

    reference_aligned = reference_aligned[feature_columns]
    current_aligned = current_aligned[feature_columns]

    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(reference_aligned[column]):
            current_aligned[column] = pd.to_numeric(
                current_aligned[column],
                errors="coerce",
            )

    return reference_aligned, current_aligned


def build_column_mapping(reference_df: pd.DataFrame) -> ColumnMapping:
    numerical_features = []
    categorical_features = []

    for column in reference_df.columns:
        if pd.api.types.is_numeric_dtype(reference_df[column]):
            numerical_features.append(column)
        else:
            categorical_features.append(column)

    return ColumnMapping(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )


def extract_drift_summary(report_dict: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "drift_detected": False,
        "drift_score": 0.0,
        "drifted_columns": 0,
        "total_columns": 0,
        "columns": {},
    }

    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})

        if "dataset_drift" in result:
            summary["drift_detected"] = bool(result.get("dataset_drift", False))
            summary["drift_score"] = float(
                result.get(
                    "share_of_drifted_columns",
                    result.get("drift_share", 0.0),
                )
                or 0.0
            )
            summary["drifted_columns"] = int(
                result.get("number_of_drifted_columns", 0) or 0
            )
            summary["total_columns"] = int(result.get("number_of_columns", 0) or 0)

        drift_by_columns = result.get("drift_by_columns", {})

        if drift_by_columns:
            for column_name, column_result in drift_by_columns.items():
                summary["columns"][column_name] = {
                    "drift_detected": column_result.get("drift_detected"),
                    "drift_score": column_result.get("drift_score"),
                    "stattest_name": column_result.get("stattest_name"),
                    "current_small_hist": column_result.get("current_small_hist"),
                    "ref_small_hist": column_result.get("ref_small_hist"),
                }

    return summary


def run_drift_check(
    model_record,
    current_dataset_id: Optional[str] = None,
    current_data: Optional[List[Dict[str, Any]]] = None,
    max_rows: int = 1000,
) -> Dict[str, Any]:
    reference_df = load_reference_data(model_record.model_id)

    current_df = build_current_dataframe(
        current_dataset_id=current_dataset_id,
        current_data=current_data,
    )

    if len(current_df) < 2:
        raise ValueError("Current data must contain at least 2 rows.")

    if len(current_df) > max_rows:
        current_df = current_df.sample(max_rows, random_state=42)

    feature_columns = model_record.feature_columns or []

    if not feature_columns:
        raise ValueError("Model does not have saved feature_columns.")

    reference_df, current_df = align_columns(
        reference_df=reference_df,
        current_df=current_df,
        feature_columns=feature_columns,
    )

    column_mapping = build_column_mapping(reference_df)

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    report_dict = report.as_dict()
    summary = extract_drift_summary(report_dict)

    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_id = generate_id("drift")
    report_path = DRIFT_REPORT_DIR / f"{report_id}.html"

    report.save_html(str(report_path))

    return {
        "report_id": report_id,
        "model_id": model_record.model_id,
        "experiment_id": model_record.experiment_id,
        "current_dataset_id": current_dataset_id,
        "drift_detected": summary["drift_detected"],
        "drift_score": summary["drift_score"],
        "drifted_columns": summary["drifted_columns"],
        "total_columns": summary["total_columns"],
        "summary": summary,
        "report_path": str(report_path),
        "html_report_url": f"/monitoring/drift/{report_id}/html",
    }
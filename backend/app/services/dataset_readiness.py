from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IDENTIFIER_HINTS = {
    "id",
    "uuid",
    "guid",
    "key",
    "code",
    "number",
    "no",
    "index",
    "serial",
}


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _column_name_looks_like_identifier(column_name: str) -> bool:
    normalized = str(column_name).strip().lower().replace("-", "_")
    pieces = {piece for piece in normalized.split("_") if piece}

    if normalized in IDENTIFIER_HINTS:
        return True

    return bool(pieces & IDENTIFIER_HINTS)


def _status_from_score(score: int, blockers: list[dict]) -> str:
    if blockers:
        return "needs_attention"
    if score >= 85:
        return "ready"
    if score >= 65:
        return "review"
    return "needs_attention"


def analyze_dataset_readiness(
    csv_path: str | Path,
    *,
    target_column: str | None = None,
) -> dict:
    """
    Produce a domain-expert-friendly dataset readiness report.

    This function does not mutate the uploaded CSV. It explains what the
    existing training pipeline can handle automatically and identifies only
    the small number of decisions that require user confirmation.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    df = pd.read_csv(path)
    rows = int(df.shape[0])
    columns = int(df.shape[1])
    target_name = str(target_column or "")
    target_is_valid = bool(target_name) and target_name in df.columns

    # Missing target values are handled separately below. This list is only
    # about input columns, so the user-facing message remains accurate.
    missing_counts = {
        str(column): int(value)
        for column, value in df.isna().sum().items()
        if int(value) > 0 and str(column) != target_name
    }
    total_missing = int(sum(missing_counts.values()))
    total_cells = max(1, rows * max(1, columns))
    missing_share = total_missing / total_cells

    duplicate_rows = int(df.duplicated().sum())
    duplicate_share = duplicate_rows / max(1, rows)

    constant_columns = [
        str(column)
        for column in df.columns
        if int(df[column].nunique(dropna=False)) <= 1
        and str(column) != target_name
    ]

    id_like_columns: list[str] = []
    high_cardinality_columns: list[str] = []

    for column in df.columns:
        column_name = str(column)
        if column_name == target_name:
            continue

        non_missing = df[column].dropna()
        if non_missing.empty:
            continue

        unique_count = int(non_missing.nunique())
        unique_ratio = unique_count / max(1, len(non_missing))
        dtype = str(df[column].dtype)

        if unique_ratio >= 0.95 and (
            _column_name_looks_like_identifier(column_name)
            or dtype.startswith("object")
            or dtype.startswith("int")
        ):
            id_like_columns.append(column_name)

        if dtype.startswith("object") and unique_count > 20 and unique_ratio >= 0.5:
            high_cardinality_columns.append(column_name)

    blockers: list[dict] = []
    auto_actions: list[dict] = []
    decisions: list[dict] = []
    notes: list[dict] = []

    if rows < 10:
        blockers.append(
            {
                "code": "too_few_rows",
                "title": "More examples are required",
                "message": (
                    "This file has fewer than 10 records. Add more records "
                    "before training so the model has enough examples to learn from."
                ),
            }
        )

    if columns < 2:
        blockers.append(
            {
                "code": "too_few_columns",
                "title": "A result column and at least one input are required",
                "message": (
                    "The dataset must contain at least one input column and one "
                    "column containing the result to predict."
                ),
            }
        )

    if not target_column:
        notes.append(
            {
                "code": "choose_target",
                "title": "Choose what you want to predict",
                "message": (
                    "Select the result column in the Data Loader to complete the "
                    "readiness check."
                ),
            }
        )
    elif target_column not in df.columns:
        blockers.append(
            {
                "code": "target_missing",
                "title": "Selected result column was not found",
                "message": (
                    f"The selected result column '{target_column}' is not present "
                    "in this dataset."
                ),
            }
        )
    else:
        target_missing = int(df[target_column].isna().sum())
        usable_target_rows = rows - target_missing

        if target_missing:
            auto_actions.append(
                {
                    "code": "drop_empty_target",
                    "title": "Ignore records without the result",
                    "message": (
                        f"{target_missing} record(s) do not contain '{target_column}'. "
                        "They will be ignored during training automatically."
                    ),
                }
            )

        if usable_target_rows < 10:
            blockers.append(
                {
                    "code": "too_few_target_rows",
                    "title": "Not enough records contain the result",
                    "message": (
                        "After ignoring records without the selected result, fewer "
                        "than 10 usable records remain."
                    ),
                }
            )

        target_values = df[target_column].dropna()
        target_unique = int(target_values.nunique())
        is_probable_classification = (
            str(target_values.dtype).startswith("object")
            or str(target_values.dtype).startswith("bool")
            or target_unique <= min(20, max(2, int(len(target_values) * 0.1)))
        )

        if is_probable_classification and target_unique >= 2:
            counts = target_values.value_counts()
            smallest = int(counts.min())
            largest = int(counts.max())
            imbalance_ratio = largest / max(1, smallest)

            if imbalance_ratio >= 4:
                notes.append(
                    {
                        "code": "uneven_outcomes",
                        "title": "One result occurs much less often",
                        "message": (
                            "AutoML Forge will use an evaluation score that gives "
                            "better visibility to less-common outcomes."
                        ),
                    }
                )

    if missing_counts:
        auto_actions.append(
            {
                "code": "fill_missing_inputs",
                "title": "Complete missing input information safely",
                "message": (
                    f"{total_missing} missing input value(s) will be handled "
                    "automatically during training."
                ),
            }
        )

    categorical_columns = [
        str(column)
        for column in df.select_dtypes(include=["object", "category", "bool"]).columns
        if str(column) != target_name
    ]

    if categorical_columns:
        auto_actions.append(
            {
                "code": "encode_categories",
                "title": "Prepare text choices for the model",
                "message": (
                    f"{len(categorical_columns)} text or category column(s) will "
                    "be converted automatically into model-ready information."
                ),
            }
        )

    # Never offer column-exclusion recommendations until the result column has
    # been selected and validated. Otherwise a high-cardinality output column
    # could be mistaken for an identifier before the assistant knows it is the
    # prediction target.
    if target_is_valid:
        for column in constant_columns:
            decisions.append(
                {
                    "code": "exclude_constant_column",
                    "action": "exclude_column",
                    "column": column,
                    "recommended": True,
                    "title": f"Exclude '{column}'",
                    "message": (
                        "This column contains the same value in every record, so it "
                        "cannot help the prediction."
                    ),
                }
            )

        for column in id_like_columns:
            if column in constant_columns:
                continue
            decisions.append(
                {
                    "code": "exclude_identifier_column",
                    "action": "exclude_column",
                    "column": column,
                    "recommended": True,
                    "title": f"Exclude '{column}'",
                    "message": (
                        "This column appears to identify individual records rather than "
                        "describe them, so excluding it is usually safer."
                    ),
                }
            )

    if duplicate_rows:
        decisions.append(
            {
                "code": "remove_duplicates",
                "action": "remove_duplicates",
                "recommended": duplicate_share <= 0.25,
                "title": "Remove repeated records",
                "message": (
                    f"{duplicate_rows} completely repeated record(s) were found. "
                    "Remove them when they are accidental copies."
                ),
            }
        )

    if high_cardinality_columns:
        notes.append(
            {
                "code": "many_categories",
                "title": "Some text fields contain many different values",
                "message": (
                    "AutoML Forge can process them, but training may take longer. "
                    "Open advanced details to review the columns."
                ),
            }
        )

    score = 100
    score -= min(20, round(missing_share * 100))
    score -= min(12, round(duplicate_share * 40))
    if target_is_valid:
        score -= min(12, len(id_like_columns) * 4)
        score -= min(10, len(constant_columns) * 3)

    if rows < 50:
        score -= 12
    elif rows < 100:
        score -= 6

    if blockers:
        score = min(score, 55)

    score = max(0, min(100, int(score)))
    status = _status_from_score(score, blockers)

    if blockers:
        summary = "A small issue must be fixed before training."
    elif decisions:
        summary = "The dataset can be trained after reviewing a few simple choices."
    elif not target_column:
        summary = "Select what you want to predict to finish the check."
    else:
        summary = "The dataset is ready. AutoML Forge can handle the technical preparation."

    return {
        "score": score,
        "status": status,
        "ready_to_train": bool(target_column) and not blockers,
        "summary": summary,
        "rows": rows,
        "columns": columns,
        "auto_actions": auto_actions,
        "decisions": decisions,
        "notes": notes,
        "blockers": blockers,
        "advanced": {
            "missing_values": missing_counts,
            "total_missing_values": total_missing,
            "duplicate_rows": duplicate_rows,
            "constant_columns": constant_columns,
            "identifier_like_columns": id_like_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
            "unique_values": {
                str(column): int(df[column].nunique(dropna=True))
                for column in df.columns
            },
            "sample": [
                {str(key): _plain_value(value) for key, value in row.items()}
                for row in df.head(5).replace({np.nan: None}).to_dict(orient="records")
            ],
        },
    }

# app/services/prediction_validation.py

from typing import Any, Dict, List, Tuple

import pandas as pd


class PredictionSchemaValidationError(Exception):
    def __init__(self, errors: List[str], warnings: List[str] | None = None):
        self.errors = errors
        self.warnings = warnings or []
        super().__init__("Prediction input schema validation failed.")


def build_input_schema(model_record) -> dict:
    feature_columns = model_record.feature_columns or []
    numerical_columns = model_record.numerical_columns or []
    categorical_columns = model_record.categorical_columns or []

    return {
        "required_features": feature_columns,
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "example": build_example_input(
            feature_columns=feature_columns,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        ),
    }


def build_example_input(
    feature_columns: List[str],
    numerical_columns: List[str],
    categorical_columns: List[str],
) -> dict:
    numerical_set = set(numerical_columns)
    categorical_set = set(categorical_columns)

    example = {}

    for column in feature_columns:
        if column in numerical_set:
            example[column] = get_numeric_example(column)

        elif column in categorical_set:
            example[column] = get_categorical_example(column)

        else:
            example[column] = "sample"

    return example


def get_numeric_example(column: str):
    key = column.lower()

    if "age" in key:
        return 30

    if "area" in key or "sqft" in key or "size" in key:
        return 1200

    if "bedroom" in key or key == "bed":
        return 3

    if "bathroom" in key or key == "bath":
        return 2

    if "fare" in key:
        return 50

    if "tenure" in key:
        return 12

    if "spend" in key or "amount" in key or "salary" in key or "income" in key:
        return 75

    if "ticket" in key or "count" in key or "quantity" in key:
        return 1

    if "class" in key or key == "pclass":
        return 3

    return 0


def get_categorical_example(column: str):
    key = column.lower()

    if "sex" in key or "gender" in key:
        return "female"

    if "location" in key or "city" in key:
        return "Urban"

    if "embarked" in key:
        return "S"

    if "plan" in key:
        return "Premium"

    if "region" in key:
        return "East"

    if "type" in key:
        return "standard"

    if "category" in key:
        return "A"

    return "sample"


def validate_and_prepare_prediction_input(
    model_record,
    input_data: Any,
) -> Tuple[pd.DataFrame, Any, dict]:
    feature_columns = model_record.feature_columns or []
    numerical_columns = model_record.numerical_columns or []
    categorical_columns = model_record.categorical_columns or []

    if not feature_columns:
        raise PredictionSchemaValidationError(
            errors=[
                "This model does not have a saved feature schema. "
                "Retrain the model to enable schema validation."
            ]
        )

    rows, is_batch = normalize_raw_input(input_data)

    errors = []
    warnings = []
    cleaned_rows = []

    feature_set = set(feature_columns)
    numerical_set = set(numerical_columns)
    categorical_set = set(categorical_columns)

    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"Row {row_index}: Input row must be a JSON object.")
            continue

        extra_columns = sorted(set(row.keys()) - feature_set)

        if extra_columns:
            warnings.append(
                f"Row {row_index}: Extra fields ignored: {', '.join(extra_columns)}."
            )

        cleaned_row = {}

        for column in feature_columns:
            value = row.get(column)

            if value is None or value == "":
                errors.append(f"Row {row_index}: Missing required feature '{column}'.")
                continue

            if column in numerical_set:
                converted_value = convert_numeric_value(
                    value=value,
                    row_index=row_index,
                    column=column,
                    errors=errors,
                )
                cleaned_row[column] = converted_value

            elif column in categorical_set:
                cleaned_row[column] = str(value)

            else:
                cleaned_row[column] = value

        cleaned_rows.append(cleaned_row)

    if errors:
        raise PredictionSchemaValidationError(
            errors=errors,
            warnings=warnings,
        )

    input_df = pd.DataFrame(cleaned_rows)

    input_df = input_df[feature_columns]

    cleaned_input = cleaned_rows if is_batch else cleaned_rows[0]

    validation_summary = {
        "valid": True,
        "mode": "batch" if is_batch else "single",
        "rows_received": len(cleaned_rows),
        "required_features": feature_columns,
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "warnings": warnings,
    }

    return input_df, cleaned_input, validation_summary


def normalize_raw_input(input_data: Any) -> tuple[list, bool]:
    if isinstance(input_data, dict):
        return [input_data], False

    if isinstance(input_data, list):
        if not input_data:
            raise PredictionSchemaValidationError(
                errors=["Input data list cannot be empty."]
            )

        return input_data, True

    raise PredictionSchemaValidationError(
        errors=["Input data must be either a JSON object or a list of JSON objects."]
    )


def convert_numeric_value(
    value: Any,
    row_index: int,
    column: str,
    errors: List[str],
):
    if isinstance(value, bool):
        errors.append(f"Row {row_index}: Feature '{column}' must be numeric.")
        return None

    try:
        converted = pd.to_numeric(value)

        if pd.isna(converted):
            errors.append(f"Row {row_index}: Feature '{column}' must be numeric.")
            return None

        return float(converted)

    except Exception:
        errors.append(f"Row {row_index}: Feature '{column}' must be numeric.")
        return None
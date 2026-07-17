from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MAX_EXPLAINED_FEATURES = 40
TOP_FACTORS = 6


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _unavailable(reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "method": "local_counterfactual_sensitivity",
        "factors": [],
    }


def _load_reference_data(reference_path: str | None, model_path: str) -> pd.DataFrame:
    if not reference_path:
        raise FileNotFoundError("The model package has no reference dataset path.")

    raw_path = Path(reference_path)
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / raw_path,
                Path(model_path).resolve().parent / raw_path,
            ]
        )

    path = next((candidate for candidate in candidates if candidate.exists()), None)

    if path is None:
        raise FileNotFoundError("The model reference dataset is no longer available.")

    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle", ".joblib"}:
        value = joblib.load(path)
        if isinstance(value, pd.DataFrame):
            return value
        return pd.DataFrame(value)

    # Older project versions may have saved a CSV without a conventional suffix.
    try:
        return pd.read_csv(path)
    except Exception:
        value = joblib.load(path)
        if isinstance(value, pd.DataFrame):
            return value
        return pd.DataFrame(value)


def _build_reference_row(
    reference_df: pd.DataFrame,
    feature_columns: list[str],
    numerical_columns: set[str],
) -> dict[str, Any]:
    reference_row: dict[str, Any] = {}

    for feature in feature_columns:
        if feature not in reference_df.columns:
            reference_row[feature] = 0 if feature in numerical_columns else "Unknown"
            continue

        series = reference_df[feature].dropna()

        if feature in numerical_columns:
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            reference_row[feature] = (
                float(numeric_series.median()) if not numeric_series.empty else 0.0
            )
        else:
            mode = series.mode(dropna=True)
            reference_row[feature] = (
                _json_safe(mode.iloc[0]) if not mode.empty else "Unknown"
            )

    return reference_row


def _probability_matrix(pipeline, rows: pd.DataFrame) -> tuple[np.ndarray, list[Any]]:
    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba(rows), dtype=float)
        classes = list(getattr(pipeline, "classes_", range(probabilities.shape[1])))
        return probabilities, classes

    if hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(rows), dtype=float)

        if scores.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-scores))
            probabilities = np.column_stack([1.0 - positive, positive])
        else:
            shifted = scores - np.max(scores, axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)

        classes = list(getattr(pipeline, "classes_", range(probabilities.shape[1])))
        return probabilities, classes

    raise AttributeError("This classifier does not expose probability-like scores.")


def _classification_factors(
    *,
    pipeline,
    rows: pd.DataFrame,
    predicted_value: Any,
    features: list[str],
    input_row: dict[str, Any],
    reference_row: dict[str, Any],
    target_column: str,
) -> tuple[list[dict], dict]:
    probabilities, classes = _probability_matrix(pipeline, rows)

    class_index = next(
        (
            index
            for index, class_value in enumerate(classes)
            if class_value == predicted_value
            or str(class_value) == str(predicted_value)
        ),
        int(np.argmax(probabilities[0])),
    )

    actual_probability = float(probabilities[0, class_index])
    factors = []

    for row_index, feature in enumerate(features, start=1):
        counterfactual_probability = float(probabilities[row_index, class_index])
        impact = actual_probability - counterfactual_probability

        direction = "increase" if impact > 0 else "decrease" if impact < 0 else "neutral"
        percentage_points = abs(impact) * 100.0
        verb = "made" if direction != "neutral" else "did not materially change"

        if direction == "increase":
            message = (
                f"{feature} made {target_column} = {predicted_value} about "
                f"{percentage_points:.1f} percentage points more likely than its typical value."
            )
        elif direction == "decrease":
            message = (
                f"{feature} made {target_column} = {predicted_value} about "
                f"{percentage_points:.1f} percentage points less likely than its typical value."
            )
        else:
            message = f"{feature} did not materially change this result."

        factors.append(
            {
                "feature": feature,
                "input_value": _json_safe(input_row.get(feature)),
                "reference_value": _json_safe(reference_row.get(feature)),
                "direction": direction,
                "impact": round(float(impact), 6),
                "impact_percentage_points": round(float(percentage_points), 2),
                "message": message,
            }
        )

    return factors, {
        "predicted_class": _json_safe(predicted_value),
        "predicted_probability": round(actual_probability, 6),
    }


def _regression_factors(
    *,
    predictions: np.ndarray,
    features: list[str],
    input_row: dict[str, Any],
    reference_row: dict[str, Any],
    target_column: str,
) -> tuple[list[dict], dict]:
    actual_prediction = float(np.ravel(predictions)[0])
    factors = []

    for row_index, feature in enumerate(features, start=1):
        counterfactual_prediction = float(np.ravel(predictions)[row_index])
        impact = actual_prediction - counterfactual_prediction
        direction = "increase" if impact > 0 else "decrease" if impact < 0 else "neutral"

        if direction == "increase":
            message = (
                f"{feature} increased the predicted {target_column} by about "
                f"{abs(impact):,.3g} compared with its typical value."
            )
        elif direction == "decrease":
            message = (
                f"{feature} reduced the predicted {target_column} by about "
                f"{abs(impact):,.3g} compared with its typical value."
            )
        else:
            message = f"{feature} did not materially change this prediction."

        factors.append(
            {
                "feature": feature,
                "input_value": _json_safe(input_row.get(feature)),
                "reference_value": _json_safe(reference_row.get(feature)),
                "direction": direction,
                "impact": round(float(impact), 6),
                "message": message,
            }
        )

    return factors, {"predicted_value": round(actual_prediction, 6)}


def build_local_explanation(
    *,
    model_path: str,
    cleaned_input: Any,
    predicted_value: Any,
    problem_type: str,
    target_column: str,
) -> dict:
    """
    Explain one prediction by replacing one feature at a time with a typical
    value learned from the training reference data.

    The function intentionally returns an unavailable payload instead of raising,
    so prediction requests never fail merely because explanation is unsupported.
    """

    try:
        if isinstance(cleaned_input, list):
            if len(cleaned_input) != 1:
                return _unavailable(
                    "Local explanations are currently available for one record at a time."
                )
            input_row = dict(cleaned_input[0])
        elif isinstance(cleaned_input, dict):
            input_row = dict(cleaned_input)
        else:
            return _unavailable("The prediction input could not be interpreted.")

        model_package = joblib.load(model_path)
        pipeline = model_package.get("pipeline")

        if pipeline is None:
            return _unavailable("The saved model package has no prediction pipeline.")

        feature_columns = list(
            model_package.get("feature_columns") or list(input_row.keys())
        )
        feature_columns = [
            feature for feature in feature_columns if feature in input_row
        ]

        if not feature_columns:
            return _unavailable("No explainable input columns were found.")

        reference_df = _load_reference_data(
            model_package.get("reference_data_path"),
            model_path,
        )
        numerical_columns = set(model_package.get("numerical_columns") or [])
        reference_row = _build_reference_row(
            reference_df,
            feature_columns,
            numerical_columns,
        )

        features = feature_columns[:MAX_EXPLAINED_FEATURES]
        original = {feature: input_row.get(feature) for feature in feature_columns}
        counterfactual_rows = [original]

        for feature in features:
            changed_row = dict(original)
            changed_row[feature] = reference_row.get(feature)
            counterfactual_rows.append(changed_row)

        rows = pd.DataFrame(counterfactual_rows, columns=feature_columns)

        if problem_type == "classification":
            factors, prediction_details = _classification_factors(
                pipeline=pipeline,
                rows=rows,
                predicted_value=predicted_value,
                features=features,
                input_row=input_row,
                reference_row=reference_row,
                target_column=target_column,
            )
        else:
            predictions = np.asarray(pipeline.predict(rows))
            factors, prediction_details = _regression_factors(
                predictions=predictions,
                features=features,
                input_row=input_row,
                reference_row=reference_row,
                target_column=target_column,
            )

        factors.sort(key=lambda item: abs(float(item.get("impact", 0.0))), reverse=True)
        strongest = abs(float(factors[0].get("impact", 0.0))) if factors else 0.0

        for factor in factors:
            factor["relative_strength"] = (
                round(abs(float(factor.get("impact", 0.0))) / strongest, 4)
                if strongest > 0
                else 0.0
            )

        visible_factors = factors[:TOP_FACTORS]
        meaningful = [
            factor for factor in visible_factors if factor["direction"] != "neutral"
        ]

        if meaningful:
            names = [factor["feature"] for factor in meaningful[:3]]
            summary = (
                f"The strongest influences on this result were "
                f"{', '.join(names)}."
            )
        else:
            summary = (
                "No single input had a strong effect when compared with typical values."
            )

        return {
            "available": True,
            "method": "local_counterfactual_sensitivity",
            "title": "Why this result was produced",
            "summary": summary,
            "problem_type": problem_type,
            "target_column": target_column,
            "prediction": _json_safe(predicted_value),
            "prediction_details": prediction_details,
            "factors": visible_factors,
            "features_evaluated": len(features),
            "features_total": len(feature_columns),
            "limited": len(feature_columns) > len(features),
            "note": (
                "Each factor is estimated by replacing only that input with a typical "
                "training value while keeping the other inputs unchanged. These are local "
                "influences, not proof of cause and effect."
            ),
        }

    except Exception as error:
        return _unavailable(
            "The prediction succeeded, but an explanation could not be generated for "
            f"this model: {error}"
        )

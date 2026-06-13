import joblib
import numpy as np
import pandas as pd

from app.services.storage import get_model_path


def load_model_package(model_id: str) -> dict:
    model_path = get_model_path(model_id)
    return joblib.load(model_path)


def make_prediction(model_id: str, input_data):
    model_package = load_model_package(model_id)

    pipeline = model_package["pipeline"]
    feature_columns = model_package["feature_columns"]

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_df = pd.DataFrame(input_data)
    else:
        raise ValueError(
            "Input data must be a dictionary or a list of dictionaries."
        )

    for column in feature_columns:
        if column not in input_df.columns:
            input_df[column] = np.nan

    input_df = input_df[feature_columns]

    predictions = pipeline.predict(input_df)

    return {
        "model_id": model_id,
        "model_name": model_package["model_name"],
        "problem_type": model_package["problem_type"],
        "target_column": model_package["target_column"],
        "prediction": predictions.tolist(),
    }
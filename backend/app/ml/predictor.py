# app/ml/predictor.py

from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.db import models
from app.db.database import Base, SessionLocal, engine
from app.services.storage import generate_id, get_model_path


def load_model_package(model_id: str) -> dict:
    model_path = get_model_path(model_id)
    return joblib.load(model_path)


def save_prediction_log_to_database(
    model_id: str,
    input_data: Any,
    prediction: list,
) -> None:
    db = SessionLocal()

    try:
        Base.metadata.create_all(bind=engine)

        prediction_record = models.PredictionLog(
            prediction_id=generate_id("prediction"),
            model_id=model_id,
            input_data=input_data,
            prediction=prediction,
        )

        db.add(prediction_record)
        db.commit()

    except Exception as error:
        db.rollback()
        print(f"PREDICTION LOG SAVE FAILED: {error}")

    finally:
        db.close()


def make_prediction(model_id: str, input_data, log_prediction: bool = True):
    model_package = load_model_package(model_id)

    pipeline = model_package["pipeline"]
    feature_columns = model_package["feature_columns"]

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])

    elif isinstance(input_data, list):
        input_df = pd.DataFrame(input_data)

    else:
        raise ValueError("Input data must be a dictionary or a list of dictionaries.")

    for column in feature_columns:
        if column not in input_df.columns:
            input_df[column] = np.nan

    input_df = input_df[feature_columns]

    predictions = pipeline.predict(input_df)
    prediction_list = predictions.tolist()

    response = {
        "model_id": model_id,
        "model_name": model_package["model_name"],
        "problem_type": model_package["problem_type"],
        "target_column": model_package["target_column"],
        "prediction": prediction_list,
    }

    if log_prediction:
        save_prediction_log_to_database(
            model_id=model_id,
            input_data=input_data,
            prediction=prediction_list,
        )

    return response
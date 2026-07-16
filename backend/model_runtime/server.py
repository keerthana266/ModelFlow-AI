import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.joblib")
model_package = None


class PredictRequest(BaseModel):
    data: dict | list[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_package

    model_package = joblib.load(MODEL_PATH)
    if "pipeline" not in model_package:
        raise RuntimeError("Model package does not contain a prediction pipeline.")

    yield
    model_package = None


app = FastAPI(
    title="AutoML Forge Model Runtime",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_id": model_package.get("model_id"),
    }


@app.get("/metadata")
def metadata():
    return {
        "model_id": model_package.get("model_id"),
        "model_name": model_package.get("model_name"),
        "problem_type": model_package.get("problem_type"),
        "target_column": model_package.get("target_column"),
        "feature_columns": model_package.get("feature_columns"),
    }


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        rows = request.data if isinstance(request.data, list) else [request.data]
        feature_columns = model_package["feature_columns"]

        missing_columns = [
            column
            for column in feature_columns
            if any(column not in row for row in rows)
        ]
        if missing_columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Missing required input columns.",
                    "columns": sorted(set(missing_columns)),
                },
            )

        input_df = pd.DataFrame(rows)[feature_columns]
        predictions = model_package["pipeline"].predict(input_df).tolist()

        return {
            "model_id": model_package["model_id"],
            "predictions": predictions,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

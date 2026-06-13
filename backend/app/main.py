from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.ml.predictor import make_prediction
from app.ml.profiler import profile_csv_dataset
from app.ml.trainer import train_automl
from app.schemas.requests import PredictRequest, TrainRequest
from app.services.storage import (
    create_storage_folders,
    get_dataset_path,
    load_experiment_result,
    save_experiment_result,
    save_uploaded_dataset,
)


app = FastAPI(
    title="AutoML Forge API",
    description="Sprint 1 backend for a no-code AutoML platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    create_storage_folders()


@app.get("/")
def root():
    return {
        "message": "AutoML Forge API is running.",
        "docs_url": "/docs",
    }


@app.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...)):
    try:
        dataset_info = save_uploaded_dataset(file)
        profile = profile_csv_dataset(get_dataset_path(dataset_info["dataset_id"]))

        return {
            "message": "Dataset uploaded successfully.",
            "dataset": {
                **dataset_info,
                "rows": profile["rows"],
                "columns": profile["columns"],
                "column_names": profile["column_names"],
            },
        }

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/datasets/{dataset_id}/profile")
def get_dataset_profile(dataset_id: str):
    try:
        dataset_path = get_dataset_path(dataset_id)
        profile = profile_csv_dataset(dataset_path)

        return {
            "dataset_id": dataset_id,
            **profile,
        }

    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.post("/experiments/start")
def start_experiment(request: TrainRequest):
    try:
        dataset_path = get_dataset_path(request.dataset_id)

        result = train_automl(
            dataset_path=dataset_path,
            target_column=request.target_column,
        )

        experiment_file = save_experiment_result(
            result["experiment_id"],
            result,
        )

        return {
            "message": "Experiment completed successfully.",
            "experiment_file": str(experiment_file),
            "result": result,
        }

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/experiments/{experiment_id}/leaderboard")
def get_experiment_leaderboard(experiment_id: str):
    try:
        result = load_experiment_result(experiment_id)

        return {
            "experiment_id": experiment_id,
            "problem_type": result["problem_type"],
            "target_column": result["target_column"],
            "main_metric": result["main_metric"],
            "best_model_id": result["best_model_id"],
            "leaderboard": result["leaderboard"],
        }

    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.get("/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    try:
        return load_experiment_result(experiment_id)

    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.post("/models/{model_id}/predict")
def predict_with_model(model_id: str, request: PredictRequest):
    try:
        return make_prediction(
            model_id=model_id,
            input_data=request.data,
        )

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
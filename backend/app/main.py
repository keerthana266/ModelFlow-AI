# app/main.py

import joblib
import numpy as np
import pandas as pd

import asyncio
from typing import Optional
from pathlib import Path
from fastapi.responses import FileResponse

from celery.result import AsyncResult
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from app.core.config import settings
from app.db import models
from app.db.database import Base, engine, get_db
from app.ml.predictor import make_prediction
from app.ml.profiler import profile_csv_dataset
from app.ml.trainer import train_automl
from app.ml.transpiler import execute_pipeline_graph
from app.schemas.requests import DriftCheckRequest
from app.ml.drift import run_drift_check
from app.schemas.requests import (
    PredictRequest,
    PipelineExecutionRequest,
    TrainRequest,
)
from app.services.storage import (
    create_storage_folders,
    delete_dataset_file,
    get_dataset_path,
    load_experiment_result,
    save_uploaded_dataset,
    generate_id,
)
from app.worker import celery_app

from app.schemas.requests import (
    DeploymentCreateRequest,
    DeploymentStatusUpdateRequest,
    DriftCheckRequest,
    PipelineExecutionRequest,
    PredictRequest,
    TrainRequest,
)


app = FastAPI(
    title="AutoML Forge API",
    description="Production-ready engine for a no-code AutoML platform.",
    version="1.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_leaderboard(leaderboard):
    cleaned_leaderboard = []

    for item in leaderboard or []:
        cleaned_item = dict(item)
        cleaned_item.pop("model_path", None)
        cleaned_leaderboard.append(cleaned_item)

    return cleaned_leaderboard


def make_json_safe(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    return value


def run_prediction_from_model_record(model_record, input_data, db: Session):
    model_package = joblib.load(model_record.model_path)

    feature_columns = model_package.get("feature_columns", [])
    pipeline = model_package["pipeline"]

    if isinstance(input_data, list):
        input_df = pd.DataFrame(input_data)
    else:
        input_df = pd.DataFrame([input_data])

    for column in feature_columns:
        if column not in input_df.columns:
            input_df[column] = None

    input_df = input_df[feature_columns]

    predictions = pipeline.predict(input_df)
    predictions = make_json_safe(predictions)

    if isinstance(predictions, list) and len(predictions) == 1:
        final_prediction = predictions[0]
    else:
        final_prediction = predictions

    prediction_id = generate_id("prediction")

    prediction_record = models.PredictionLog(
        prediction_id=prediction_id,
        model_id=model_record.model_id,
        input_data=make_json_safe(input_data),
        prediction=make_json_safe(final_prediction),
    )

    db.add(prediction_record)
    db.commit()

    return {
        "prediction_id": prediction_id,
        "model_id": model_record.model_id,
        "model_name": model_record.model_name,
        "problem_type": model_record.problem_type,
        "target_column": model_record.target_column,
        "prediction": final_prediction,
    }

@app.on_event("startup")
def startup_event():
    create_storage_folders()

    print("Creating database tables if they do not exist...")
    Base.metadata.create_all(bind=engine)

    print("Connecting to PostgreSQL database...")

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        print("DATABASE STATUS: Successfully connected to PostgreSQL!")

    except Exception as error:
        print("DATABASE STATUS: Connection FAILED!")
        print(f"Error Details: {error}")
        print("Please verify PostgreSQL and DATABASE_URL.")


@app.get("/")
def root():
    return {
        "message": "AutoML Forge API is running.",
        "docs_url": "/docs",
        "version": "1.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "automl-forge-api",
    }


@app.post("/datasets/upload")
def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        dataset_info = save_uploaded_dataset(
            file=file,
            max_upload_mb=settings.MAX_UPLOAD_MB,
        )

        dataset_path = get_dataset_path(dataset_info["dataset_id"])
        profile = profile_csv_dataset(dataset_path)

        dataset_record = models.DatasetRecord(
            dataset_id=dataset_info["dataset_id"],
            original_filename=dataset_info["original_filename"],
            file_path=dataset_info["saved_path"],
            rows=profile["rows"],
            columns=profile["columns"],
            column_names=profile["column_names"],
            dtypes=profile["dtypes"],
            missing_values=profile["missing_values"],
            unique_values=profile["unique_values"],
            duplicate_rows=profile["duplicate_rows"],
        )

        db.add(dataset_record)
        db.commit()
        db.refresh(dataset_record)

        return {
            "message": "Dataset uploaded successfully.",
            "dataset": {
                "dataset_id": dataset_record.dataset_id,
                "original_filename": dataset_record.original_filename,
                "saved_path": dataset_record.file_path,
                "rows": dataset_record.rows,
                "columns": dataset_record.columns,
                "column_names": dataset_record.column_names,
                "uploaded_at": dataset_record.uploaded_at,
            },
        }

    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    datasets = (
        db.query(models.DatasetRecord)
        .order_by(models.DatasetRecord.uploaded_at.desc())
        .all()
    )

    return {
        "count": len(datasets),
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "original_filename": dataset.original_filename,
                "rows": dataset.rows,
                "columns": dataset.columns,
                "column_names": dataset.column_names,
                "uploaded_at": dataset.uploaded_at,
            }
            for dataset in datasets
        ],
    }


@app.get("/datasets/{dataset_id}")
def get_dataset_details(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    dataset = (
        db.query(models.DatasetRecord)
        .filter(models.DatasetRecord.dataset_id == dataset_id)
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    return {
        "dataset_id": dataset.dataset_id,
        "original_filename": dataset.original_filename,
        "file_path": dataset.file_path,
        "rows": dataset.rows,
        "columns": dataset.columns,
        "column_names": dataset.column_names,
        "dtypes": dataset.dtypes,
        "missing_values": dataset.missing_values,
        "unique_values": dataset.unique_values,
        "duplicate_rows": dataset.duplicate_rows,
        "uploaded_at": dataset.uploaded_at,
    }


@app.get("/datasets/{dataset_id}/profile")
def get_dataset_profile(dataset_id: str):
    try:
        dataset_path = get_dataset_path(dataset_id)
        profile = profile_csv_dataset(dataset_path)

        return {
            "dataset_id": dataset_id,
            "profile": profile,
        }

    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.get("/datasets/{dataset_id}/columns")
def get_dataset_columns(dataset_id: str):
    try:
        dataset_path = get_dataset_path(dataset_id)
        profile = profile_csv_dataset(dataset_path)

        return {
            "dataset_id": dataset_id,
            "columns": profile["column_names"],
            "dtypes": profile["dtypes"],
        }

    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.delete("/datasets/{dataset_id}")
def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    dataset = (
        db.query(models.DatasetRecord)
        .filter(models.DatasetRecord.dataset_id == dataset_id)
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        delete_dataset_file(dataset_id)
        db.delete(dataset)
        db.commit()

        return {
            "message": "Dataset deleted successfully.",
            "dataset_id": dataset_id,
        }

    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/experiments/start")
def start_experiment(request: TrainRequest):
    try:
        dataset_path = get_dataset_path(request.dataset_id)

        task = train_automl.delay(
            dataset_path_str=str(dataset_path),
            target_column=request.target_column,
            time_budget=request.time_budget,
        )

        return {
            "message": "Experiment queued successfully.",
            "task_id": task.id,
        }

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/experiments/task/{task_id}")
def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task_result.status,
    }

    if task_result.status == "SUCCESS":
        response["result"] = task_result.result

    elif task_result.status == "FAILURE":
        response["error"] = str(task_result.result)

    elif task_result.status == "PROGRESS":
        response["details"] = task_result.info

    return response


@app.get("/experiments")
def list_experiments(db: Session = Depends(get_db)):
    experiments = (
        db.query(models.ExperimentResult)
        .order_by(models.ExperimentResult.created_at.desc())
        .all()
    )

    return {
        "count": len(experiments),
        "experiments": [
            {
                "experiment_id": experiment.experiment_id,
                "problem_type": experiment.problem_type,
                "target_column": experiment.target_column,
                "rows_used": experiment.rows_used,
                "best_model_id": experiment.best_model_id,
                "main_metric": experiment.main_metric,
                "best_score": experiment.best_score,
                "created_at": experiment.created_at,
            }
            for experiment in experiments
        ],
    }


@app.get("/experiments/{experiment_id}/leaderboard")
def get_experiment_leaderboard(
    experiment_id: str,
    db: Session = Depends(get_db),
):
    try:
        record = (
            db.query(models.ExperimentResult)
            .filter(models.ExperimentResult.experiment_id == experiment_id)
            .first()
        )

        if record:
            return {
                "experiment_id": record.experiment_id,
                "problem_type": record.problem_type,
                "target_column": record.target_column,
                "main_metric": record.main_metric,
                "best_model_id": record.best_model_id,
                "leaderboard": sanitize_leaderboard(record.leaderboard),
            }

        result = load_experiment_result(experiment_id)

        return {
            "experiment_id": result["experiment_id"],
            "problem_type": result["problem_type"],
            "target_column": result["target_column"],
            "main_metric": result["main_metric"],
            "best_model_id": result["best_model_id"],
            "leaderboard": sanitize_leaderboard(result["leaderboard"]),
        }

    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.get("/experiments/{experiment_id}")
def get_experiment_details(
    experiment_id: str,
    db: Session = Depends(get_db),
):
    try:
        record = (
            db.query(models.ExperimentResult)
            .filter(models.ExperimentResult.experiment_id == experiment_id)
            .first()
        )

        if record:
            return {
                "experiment_id": record.experiment_id,
                "problem_type": record.problem_type,
                "target_column": record.target_column,
                "rows_used": record.rows_used,
                "best_model_id": record.best_model_id,
                "main_metric": record.main_metric,
                "best_score": record.best_score,
                "feature_importance": record.feature_importance,
                "leaderboard": sanitize_leaderboard(record.leaderboard),
                "created_at": record.created_at,
            }

        result = load_experiment_result(experiment_id)
        result["leaderboard"] = sanitize_leaderboard(result.get("leaderboard", []))
        return result

    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.get("/models")
def list_models(db: Session = Depends(get_db)):
    trained_models = (
        db.query(models.TrainedModel)
        .order_by(models.TrainedModel.created_at.desc())
        .all()
    )

    return {
        "count": len(trained_models),
        "models": [
            {
                "model_id": model.model_id,
                "experiment_id": model.experiment_id,
                "model_name": model.model_name,
                "problem_type": model.problem_type,
                "target_column": model.target_column,
                "main_metric": model.main_metric,
                "score": model.score,
                "metrics": model.metrics,
                "created_at": model.created_at,
            }
            for model in trained_models
        ],
    }


@app.get("/models/{model_id}")
def get_model_details(
    model_id: str,
    db: Session = Depends(get_db),
):
    model = (
        db.query(models.TrainedModel)
        .filter(models.TrainedModel.model_id == model_id)
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    return {
        "model_id": model.model_id,
        "experiment_id": model.experiment_id,
        "model_name": model.model_name,
        "problem_type": model.problem_type,
        "target_column": model.target_column,
        "main_metric": model.main_metric,
        "score": model.score,
        "metrics": model.metrics,
        "feature_columns": model.feature_columns,
        "numerical_columns": model.numerical_columns,
        "categorical_columns": model.categorical_columns,
        "created_at": model.created_at,
    }


@app.post("/models/{model_id}/predict")
def predict_with_model(
    model_id: str,
    request: PredictRequest,
    db: Session = Depends(get_db),
):
    model_record = (
        db.query(models.TrainedModel)
        .filter(models.TrainedModel.model_id == model_id)
        .first()
    )

    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        return run_prediction_from_model_record(
            model_record=model_record,
            input_data=request.data,
            db=db,
        )

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/predictions")
def list_prediction_logs(
    model_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    query = db.query(models.PredictionLog)

    if model_id:
        query = query.filter(models.PredictionLog.model_id == model_id)

    logs = (
        query.order_by(models.PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "count": len(logs),
        "predictions": [
            {
                "prediction_id": log.prediction_id,
                "model_id": log.model_id,
                "input_data": log.input_data,
                "prediction": log.prediction,
                "created_at": log.created_at,
            }
            for log in logs
        ],
    }


@app.post("/pipeline/execute")
def execute_visual_pipeline(request: PipelineExecutionRequest):
    try:
        nodes_dict = [node.model_dump() for node in request.nodes]
        edges_dict = [edge.model_dump() for edge in request.edges]

        result = execute_pipeline_graph(nodes_dict, edges_dict)
        return result

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/pipeline/save")
def save_visual_pipeline(
    request: PipelineExecutionRequest,
    db: Session = Depends(get_db),
):
    try:
        new_graph = models.PipelineGraph(
            nodes=[node.model_dump() for node in request.nodes],
            edges=[edge.model_dump() for edge in request.edges],
        )

        db.add(new_graph)
        db.commit()
        db.refresh(new_graph)

        return {
            "message": "Visual canvas state committed successfully to PostgreSQL.",
            "graph_id": str(new_graph.id),
        }

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Database persistence failed to process transaction.",
        )


@app.get("/pipelines")
def list_saved_pipelines(db: Session = Depends(get_db)):
    pipelines = (
        db.query(models.PipelineGraph)
        .order_by(models.PipelineGraph.created_at.desc())
        .all()
    )

    return {
        "count": len(pipelines),
        "pipelines": [
            {
                "graph_id": str(pipeline.id),
                "nodes": pipeline.nodes,
                "edges": pipeline.edges,
                "created_at": pipeline.created_at,
            }
            for pipeline in pipelines
        ],
    }


@app.get("/pipelines/{graph_id}")
def get_saved_pipeline(
    graph_id: str,
    db: Session = Depends(get_db),
):
    pipeline = (
        db.query(models.PipelineGraph)
        .filter(models.PipelineGraph.id == graph_id)
        .first()
    )

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found.")

    return {
        "graph_id": str(pipeline.id),
        "nodes": pipeline.nodes,
        "edges": pipeline.edges,
        "created_at": pipeline.created_at,
    }


@app.websocket("/ws/task/{task_id}")
async def task_status_websocket(

    websocket: WebSocket,
    task_id: str,
):
    await websocket.accept()

    try:
        previous_state = None
        previous_message = None

        while True:
            task_result = AsyncResult(task_id, app=celery_app)
            current_state = task_result.state

            meta = task_result.info if isinstance(task_result.info, dict) else {}
            current_message = meta.get("message", "")

            if current_state != previous_state or current_message != previous_message:
                payload = {
                    "task_id": task_id,
                    "status": current_state,
                    "message": current_message,
                }

                if current_state == "SUCCESS":
                    payload["result"] = task_result.result

                elif current_state == "FAILURE":
                    payload["error"] = str(task_result.result)

                await websocket.send_json(payload)

                previous_state = current_state
                previous_message = current_message

            if current_state in ["SUCCESS", "FAILURE"]:
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print(f"Client disconnected from task {task_id} stream.")

@app.post("/monitoring/drift")
def create_drift_report(
    request: DriftCheckRequest,
    db: Session = Depends(get_db),
):
    model_record = (
        db.query(models.TrainedModel)
        .filter(models.TrainedModel.model_id == request.model_id)
        .first()
    )

    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        result = run_drift_check(
            model_record=model_record,
            current_dataset_id=request.current_dataset_id,
            current_data=request.current_data,
            max_rows=request.max_rows,
        )

        drift_record = models.DriftReport(
            report_id=result["report_id"],
            model_id=result["model_id"],
            experiment_id=result["experiment_id"],
            current_dataset_id=result["current_dataset_id"],
            drift_detected=result["drift_detected"],
            drift_score=result["drift_score"],
            drifted_columns=result["drifted_columns"],
            total_columns=result["total_columns"],
            report_path=result["report_path"],
            summary=result["summary"],
        )

        db.merge(drift_record)
        db.commit()

        clean_result = dict(result)
        clean_result.pop("report_path", None)

        return clean_result

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/monitoring/drift")
def list_drift_reports(
    limit: int = 20,
    db: Session = Depends(get_db),
):
    records = (
        db.query(models.DriftReport)
        .order_by(models.DriftReport.created_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "drift_reports": [
            {
                "report_id": record.report_id,
                "model_id": record.model_id,
                "experiment_id": record.experiment_id,
                "current_dataset_id": record.current_dataset_id,
                "drift_detected": record.drift_detected,
                "drift_score": record.drift_score,
                "drifted_columns": record.drifted_columns,
                "total_columns": record.total_columns,
                "html_report_url": f"/monitoring/drift/{record.report_id}/html",
                "created_at": record.created_at,
            }
            for record in records
        ]
    }


@app.get("/monitoring/drift/{report_id}")
def get_drift_report(
    report_id: str,
    db: Session = Depends(get_db),
):
    record = (
        db.query(models.DriftReport)
        .filter(models.DriftReport.report_id == report_id)
        .first()
    )

    if not record:
        raise HTTPException(status_code=404, detail="Drift report not found.")

    return {
        "report_id": record.report_id,
        "model_id": record.model_id,
        "experiment_id": record.experiment_id,
        "current_dataset_id": record.current_dataset_id,
        "drift_detected": record.drift_detected,
        "drift_score": record.drift_score,
        "drifted_columns": record.drifted_columns,
        "total_columns": record.total_columns,
        "summary": record.summary,
        "html_report_url": f"/monitoring/drift/{record.report_id}/html",
        "created_at": record.created_at,
    }


@app.get("/monitoring/drift/{report_id}/html")
def get_drift_report_html(
    report_id: str,
    db: Session = Depends(get_db),
):
    record = (
        db.query(models.DriftReport)
        .filter(models.DriftReport.report_id == report_id)
        .first()
    )

    if not record:
        raise HTTPException(status_code=404, detail="Drift report not found.")

    report_path = Path(record.report_path)

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Drift HTML file not found.")

    return FileResponse(
        path=str(report_path),
        media_type="text/html",
        filename=f"{report_id}.html",
    )

@app.post("/deployments")
def create_model_deployment(
    request: DeploymentCreateRequest,
    db: Session = Depends(get_db),
):
    model_record = (
        db.query(models.TrainedModel)
        .filter(models.TrainedModel.model_id == request.model_id)
        .first()
    )

    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found.")

    latest_deployment = (
        db.query(models.ModelDeployment)
        .filter(models.ModelDeployment.deployment_name == request.deployment_name)
        .filter(models.ModelDeployment.environment == request.environment)
        .order_by(models.ModelDeployment.version.desc())
        .first()
    )

    next_version = 1

    if latest_deployment:
        next_version = latest_deployment.version + 1

    if request.activate:
        (
            db.query(models.ModelDeployment)
            .filter(models.ModelDeployment.deployment_name == request.deployment_name)
            .filter(models.ModelDeployment.environment == request.environment)
            .filter(models.ModelDeployment.status == "active")
            .update({"status": "inactive"})
        )

    deployment_id = generate_id("deployment")
    endpoint_path = f"/deployments/{deployment_id}/predict"

    deployment = models.ModelDeployment(
        deployment_id=deployment_id,
        deployment_name=request.deployment_name,
        model_id=model_record.model_id,
        experiment_id=model_record.experiment_id,
        version=next_version,
        environment=request.environment,
        status="active" if request.activate else "inactive",
        endpoint_path=endpoint_path,
    )

    db.add(deployment)
    db.commit()
    db.refresh(deployment)

    return {
        "deployment_id": deployment.deployment_id,
        "deployment_name": deployment.deployment_name,
        "model_id": deployment.model_id,
        "experiment_id": deployment.experiment_id,
        "version": deployment.version,
        "environment": deployment.environment,
        "status": deployment.status,
        "endpoint_path": deployment.endpoint_path,
        "versioned_endpoint_path": (
            f"/deployments/name/{deployment.deployment_name}"
            f"/versions/{deployment.version}/predict"
        ),
        "created_at": deployment.created_at,
    }

@app.get("/deployments")
def list_model_deployments(
    db: Session = Depends(get_db),
):
    deployments = (
        db.query(models.ModelDeployment)
        .order_by(models.ModelDeployment.created_at.desc())
        .all()
    )

    return {
        "deployments": [
            {
                "deployment_id": deployment.deployment_id,
                "deployment_name": deployment.deployment_name,
                "model_id": deployment.model_id,
                "experiment_id": deployment.experiment_id,
                "version": deployment.version,
                "environment": deployment.environment,
                "status": deployment.status,
                "endpoint_path": deployment.endpoint_path,
                "versioned_endpoint_path": (
                    f"/deployments/name/{deployment.deployment_name}"
                    f"/versions/{deployment.version}/predict"
                ),
                "created_at": deployment.created_at,
            }
            for deployment in deployments
        ]
    }

@app.get("/deployments/{deployment_id}")
def get_model_deployment(
    deployment_id: str,
    db: Session = Depends(get_db),
):
    deployment = (
        db.query(models.ModelDeployment)
        .filter(models.ModelDeployment.deployment_id == deployment_id)
        .first()
    )

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found.")

    return {
        "deployment_id": deployment.deployment_id,
        "deployment_name": deployment.deployment_name,
        "model_id": deployment.model_id,
        "experiment_id": deployment.experiment_id,
        "version": deployment.version,
        "environment": deployment.environment,
        "status": deployment.status,
        "endpoint_path": deployment.endpoint_path,
        "versioned_endpoint_path": (
            f"/deployments/name/{deployment.deployment_name}"
            f"/versions/{deployment.version}/predict"
        ),
        "created_at": deployment.created_at,
    }

@app.patch("/deployments/{deployment_id}/status")
def update_model_deployment_status(
    deployment_id: str,
    request: DeploymentStatusUpdateRequest,
    db: Session = Depends(get_db),
):
    deployment = (
        db.query(models.ModelDeployment)
        .filter(models.ModelDeployment.deployment_id == deployment_id)
        .first()
    )

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found.")

    if request.status == "active":
        (
            db.query(models.ModelDeployment)
            .filter(models.ModelDeployment.deployment_name == deployment.deployment_name)
            .filter(models.ModelDeployment.environment == deployment.environment)
            .filter(models.ModelDeployment.status == "active")
            .update({"status": "inactive"})
        )

    deployment.status = request.status

    db.commit()
    db.refresh(deployment)

    return {
        "deployment_id": deployment.deployment_id,
        "deployment_name": deployment.deployment_name,
        "model_id": deployment.model_id,
        "version": deployment.version,
        "environment": deployment.environment,
        "status": deployment.status,
    }

@app.post("/deployments/{deployment_id}/predict")
def predict_with_deployment(
    deployment_id: str,
    request: PredictRequest,
    db: Session = Depends(get_db),
):
    deployment = (
        db.query(models.ModelDeployment)
        .filter(models.ModelDeployment.deployment_id == deployment_id)
        .first()
    )

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found.")

    if deployment.status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Deployment is not active. Current status: {deployment.status}",
        )

    model_record = (
        db.query(models.TrainedModel)
        .filter(models.TrainedModel.model_id == deployment.model_id)
        .first()
    )

    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        prediction_result = run_prediction_from_model_record(
            model_record=model_record,
            input_data=request.data,
            db=db,
        )

        prediction_result["deployment_id"] = deployment.deployment_id
        prediction_result["deployment_name"] = deployment.deployment_name
        prediction_result["deployment_version"] = deployment.version
        prediction_result["environment"] = deployment.environment

        return prediction_result

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
    

@app.post("/deployments/name/{deployment_name}/versions/{version}/predict")
def predict_with_deployment_version(
    deployment_name: str,
    version: int,
    request: PredictRequest,
    db: Session = Depends(get_db),
):
    deployment = (
        db.query(models.ModelDeployment)
        .filter(models.ModelDeployment.deployment_name == deployment_name)
        .filter(models.ModelDeployment.version == version)
        .first()
    )

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment version not found.")

    if deployment.status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Deployment version is not active. Current status: {deployment.status}",
        )

    model_record = (
        db.query(models.TrainedModel)
        .filter(models.TrainedModel.model_id == deployment.model_id)
        .first()
    )

    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        prediction_result = run_prediction_from_model_record(
            model_record=model_record,
            input_data=request.data,
            db=db,
        )

        prediction_result["deployment_id"] = deployment.deployment_id
        prediction_result["deployment_name"] = deployment.deployment_name
        prediction_result["deployment_version"] = deployment.version
        prediction_result["environment"] = deployment.environment

        return prediction_result

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
    

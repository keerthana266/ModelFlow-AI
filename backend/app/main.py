# app/main.py

import asyncio
from typing import Optional

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
)
from app.worker import celery_app


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
                "leaderboard": record.leaderboard,
            }

        result = load_experiment_result(experiment_id)

        return {
            "experiment_id": result["experiment_id"],
            "problem_type": result["problem_type"],
            "target_column": result["target_column"],
            "main_metric": result["main_metric"],
            "best_model_id": result["best_model_id"],
            "leaderboard": result["leaderboard"],
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
                "leaderboard": record.leaderboard,
                "created_at": record.created_at,
            }

        return load_experiment_result(experiment_id)

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
        "model_path": model.model_path,
        "created_at": model.created_at,
    }


@app.post("/models/{model_id}/predict")
def predict_with_model(
    model_id: str,
    request: PredictRequest,
):
    try:
        return make_prediction(
            model_id=model_id,
            input_data=request.data,
            log_prediction=True,
        )

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

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
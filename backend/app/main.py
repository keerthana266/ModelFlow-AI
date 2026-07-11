import asyncio
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
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
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from app.core.config import settings
from app.core.security import decode_access_token
from app.db import models
from app.db.database import Base, SessionLocal, engine, get_db
from app.db.migrations import ensure_project_scope_columns
from app.dependencies.auth import get_current_user
from app.dependencies.project import get_current_project
from app.ml.drift import run_drift_check
from app.ml.profiler import profile_csv_dataset
from app.ml.trainer import train_automl
from app.ml.transpiler import execute_pipeline_graph
from app.routes.auth import router as auth_router
from app.routes.projects import router as projects_router
from app.schemas.requests import (
    DeploymentCreateRequest,
    DeploymentStatusUpdateRequest,
    DriftCheckRequest,
    PipelineExecutionRequest,
    PredictRequest,
    TrainRequest,
)
from app.services.artifact_store import get_artifact_store_status
from app.services.prediction_validation import (
    PredictionSchemaValidationError,
    build_input_schema,
    validate_and_prepare_prediction_input,
)
from app.services.storage import (
    create_storage_folders,
    delete_dataset_file,
    generate_id,
    get_dataset_path,
    save_uploaded_dataset,
)
from app.worker import celery_app


app = FastAPI(
    title="AutoML Forge API",
    description="Project-scoped no-code AutoML platform API.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(projects_router)


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
        return {key: make_json_safe(item) for key, item in value.items()}

    return value


def scoped_query(db: Session, model, current_user, current_project):
    return (
        db.query(model)
        .filter(model.owner_id == current_user.user_id)
        .filter(model.project_id == current_project.project_id)
    )


def get_scoped_dataset(
    db: Session,
    dataset_id: str,
    current_user: models.User,
    current_project: models.Project,
) -> models.DatasetRecord:
    dataset = (
        scoped_query(db, models.DatasetRecord, current_user, current_project)
        .filter(models.DatasetRecord.dataset_id == dataset_id)
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found in this project.")

    return dataset


def get_scoped_model(
    db: Session,
    model_id: str,
    current_user: models.User,
    current_project: models.Project,
) -> models.TrainedModel:
    model_record = (
        scoped_query(db, models.TrainedModel, current_user, current_project)
        .filter(models.TrainedModel.model_id == model_id)
        .first()
    )

    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found in this project.")

    return model_record


def run_prediction_from_model_record(
    model_record,
    input_data,
    db: Session,
    owner_id: str,
    project_id: str,
):
    try:
        model_package = joblib.load(model_record.model_path)
        input_df, cleaned_input, validation_summary = (
            validate_and_prepare_prediction_input(
                model_record=model_record,
                input_data=input_data,
            )
        )

        predictions = model_package["pipeline"].predict(input_df)
        predictions = make_json_safe(predictions)
        final_prediction = (
            predictions[0]
            if isinstance(predictions, list) and len(predictions) == 1
            else predictions
        )

        prediction_id = generate_id("prediction")
        prediction_record = models.PredictionLog(
            prediction_id=prediction_id,
            owner_id=owner_id,
            project_id=project_id,
            model_id=model_record.model_id,
            input_data=make_json_safe(cleaned_input),
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
            "prediction": make_json_safe(final_prediction),
            "schema_validation": validation_summary,
        }

    except PredictionSchemaValidationError as error:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Prediction input schema validation failed.",
                "errors": error.errors,
                "warnings": error.warnings,
                "input_schema": build_input_schema(model_record),
            },
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.on_event("startup")
def startup_event():
    create_storage_folders()
    Base.metadata.create_all(bind=engine)
    ensure_project_scope_columns()

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("DATABASE STATUS: Successfully connected to the configured database!")
    except Exception as error:
        print("DATABASE STATUS: Connection FAILED!")
        print(f"Error Details: {error}")


@app.get("/")
def root():
    return {
        "message": "AutoML Forge API is running.",
        "docs_url": "/docs",
        "version": "2.0.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "automl-forge-api"}


@app.get("/storage/health")
def storage_health():
    return get_artifact_store_status()


@app.post("/datasets/upload")
def upload_dataset(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
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
            owner_id=current_user.user_id,
            project_id=current_project.project_id,
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
def list_datasets(
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    datasets = (
        scoped_query(db, models.DatasetRecord, current_user, current_project)
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
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = get_scoped_dataset(
        db, dataset_id, current_user, current_project
    )

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
def get_dataset_profile(
    dataset_id: str,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = get_scoped_dataset(db, dataset_id, current_user, current_project)

    try:
        return {
            "dataset_id": dataset_id,
            "profile": profile_csv_dataset(Path(dataset.file_path)),
        }
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.get("/datasets/{dataset_id}/columns")
def get_dataset_columns(
    dataset_id: str,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = get_scoped_dataset(db, dataset_id, current_user, current_project)

    try:
        profile = profile_csv_dataset(Path(dataset.file_path))
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
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = get_scoped_dataset(db, dataset_id, current_user, current_project)

    try:
        delete_dataset_file(dataset_id)
        db.delete(dataset)
        db.commit()
        return {"message": "Dataset deleted successfully.", "dataset_id": dataset_id}
    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/experiments/start")
def start_experiment(
    request: TrainRequest,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = get_scoped_dataset(
        db, request.dataset_id, current_user, current_project
    )

    try:
        task = train_automl.delay(
            str(dataset.file_path),
            request.target_column,
            request.time_budget,
            request.tuning_mode,
            current_user.user_id,
            current_project.project_id,
        )

        db.add(
            models.TaskRun(
                task_id=task.id,
                owner_id=current_user.user_id,
                project_id=current_project.project_id,
                task_type="training",
            )
        )
        db.commit()

        return {"message": "Experiment queued successfully.", "task_id": task.id}
    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/experiments/task/{task_id}")
def get_task_status(
    task_id: str,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    task_record = (
        db.query(models.TaskRun)
        .filter(models.TaskRun.task_id == task_id)
        .filter(models.TaskRun.owner_id == current_user.user_id)
        .filter(models.TaskRun.project_id == current_project.project_id)
        .first()
    )

    if not task_record:
        raise HTTPException(status_code=404, detail="Task not found in this project.")

    task_result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": task_result.status}

    if task_result.status == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.status == "FAILURE":
        response["error"] = str(task_result.result)
    elif task_result.status == "PROGRESS":
        response["details"] = task_result.info

    return response


@app.get("/experiments")
def list_experiments(
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    experiments = (
        scoped_query(db, models.ExperimentResult, current_user, current_project)
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
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    record = (
        scoped_query(db, models.ExperimentResult, current_user, current_project)
        .filter(models.ExperimentResult.experiment_id == experiment_id)
        .first()
    )

    if not record:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    return {
        "experiment_id": record.experiment_id,
        "problem_type": record.problem_type,
        "target_column": record.target_column,
        "main_metric": record.main_metric,
        "best_model_id": record.best_model_id,
        "leaderboard": sanitize_leaderboard(record.leaderboard),
    }


@app.get("/experiments/{experiment_id}")
def get_experiment_details(
    experiment_id: str,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    record = (
        scoped_query(db, models.ExperimentResult, current_user, current_project)
        .filter(models.ExperimentResult.experiment_id == experiment_id)
        .first()
    )

    if not record:
        raise HTTPException(status_code=404, detail="Experiment not found.")

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


@app.get("/models")
def list_models(
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    trained_models = (
        scoped_query(db, models.TrainedModel, current_user, current_project)
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
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    model = get_scoped_model(db, model_id, current_user, current_project)
    artifact_store = {}

    try:
        model_package = joblib.load(model.model_path)
        artifact_store = model_package.get("artifact_store", {})
    except Exception as error:
        artifact_store = {"status": "unavailable", "error": str(error)}

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
        "input_schema": build_input_schema(model),
        "artifact_store": artifact_store,
        "created_at": model.created_at,
    }


@app.post("/models/{model_id}/predict")
def predict_with_model(
    model_id: str,
    request: PredictRequest,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    model_record = get_scoped_model(
        db, model_id, current_user, current_project
    )

    return run_prediction_from_model_record(
        model_record=model_record,
        input_data=request.data,
        db=db,
        owner_id=current_user.user_id,
        project_id=current_project.project_id,
    )


@app.get("/predictions")
def list_prediction_logs(
    model_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    query = scoped_query(db, models.PredictionLog, current_user, current_project)

    if model_id:
        query = query.filter(models.PredictionLog.model_id == model_id)

    logs = query.order_by(models.PredictionLog.created_at.desc()).limit(limit).all()

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
def execute_visual_pipeline(
    request: PipelineExecutionRequest,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    nodes_dict = [node.model_dump() for node in request.nodes]
    edges_dict = [edge.model_dump() for edge in request.edges]

    data_loader_nodes = [
        node for node in nodes_dict if str(node.get("type", "")).lower() == "dataloader"
    ]

    for node in data_loader_nodes:
        dataset_id = (node.get("data") or {}).get("dataset_id")
        if dataset_id:
            get_scoped_dataset(db, dataset_id, current_user, current_project)

    try:
        result = execute_pipeline_graph(
            nodes_dict,
            edges_dict,
            owner_id=current_user.user_id,
            project_id=current_project.project_id,
        )

        task_id = result.get("final_state", {}).get("task_id")
        if task_id:
            db.merge(
                models.TaskRun(
                    task_id=task_id,
                    owner_id=current_user.user_id,
                    project_id=current_project.project_id,
                    task_type="pipeline_training",
                )
            )
            db.commit()

        return result
    except HTTPException:
        raise
    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/pipeline/save")
def save_visual_pipeline(
    request: PipelineExecutionRequest,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    try:
        new_graph = models.PipelineGraph(
            owner_id=current_user.user_id,
            project_id=current_project.project_id,
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
def list_saved_pipelines(
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    pipelines = (
        scoped_query(db, models.PipelineGraph, current_user, current_project)
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
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    pipeline = (
        scoped_query(db, models.PipelineGraph, current_user, current_project)
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
async def task_status_websocket(websocket: WebSocket, task_id: str):
    token = websocket.query_params.get("token", "")
    project_id = websocket.query_params.get("project_id", "")
    db = SessionLocal()

    try:
        payload = decode_access_token(token)
        user_id = str(payload["sub"])

        valid_task = (
            db.query(models.TaskRun)
            .filter(models.TaskRun.task_id == task_id)
            .filter(models.TaskRun.owner_id == user_id)
            .filter(models.TaskRun.project_id == project_id)
            .first()
        )

        if not valid_task:
            await websocket.close(code=4403)
            return

        await websocket.accept()
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
    except (ValueError, KeyError):
        await websocket.close(code=4401)
    except WebSocketDisconnect:
        print(f"Client disconnected from task {task_id} stream.")
    finally:
        db.close()


@app.post("/monitoring/drift")
def create_drift_report(
    request: DriftCheckRequest,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    model_record = get_scoped_model(
        db, request.model_id, current_user, current_project
    )

    if request.current_dataset_id:
        get_scoped_dataset(
            db, request.current_dataset_id, current_user, current_project
        )

    try:
        result = run_drift_check(
            model_record=model_record,
            current_dataset_id=request.current_dataset_id,
            current_data=request.current_data,
            max_rows=request.max_rows,
        )

        drift_record = models.DriftReport(
            report_id=result["report_id"],
            owner_id=current_user.user_id,
            project_id=current_project.project_id,
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
        db.rollback()
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/monitoring/drift")
def list_drift_reports(
    limit: int = Query(default=20, ge=1, le=200),
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    records = (
        scoped_query(db, models.DriftReport, current_user, current_project)
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
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    record = (
        scoped_query(db, models.DriftReport, current_user, current_project)
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
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    record = (
        scoped_query(db, models.DriftReport, current_user, current_project)
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
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    model_record = get_scoped_model(
        db, request.model_id, current_user, current_project
    )

    latest_deployment = (
        scoped_query(db, models.ModelDeployment, current_user, current_project)
        .filter(models.ModelDeployment.deployment_name == request.deployment_name)
        .order_by(models.ModelDeployment.version.desc())
        .first()
    )
    next_version = latest_deployment.version + 1 if latest_deployment else 1

    if request.activate:
        (
            scoped_query(db, models.ModelDeployment, current_user, current_project)
            .filter(models.ModelDeployment.deployment_name == request.deployment_name)
            .filter(models.ModelDeployment.environment == request.environment)
            .filter(models.ModelDeployment.status == "active")
            .update({"status": "inactive"})
        )

    deployment_id = generate_id("deployment")
    deployment = models.ModelDeployment(
        deployment_id=deployment_id,
        owner_id=current_user.user_id,
        project_id=current_project.project_id,
        deployment_name=request.deployment_name,
        model_id=model_record.model_id,
        experiment_id=model_record.experiment_id,
        version=next_version,
        environment=request.environment,
        status="active" if request.activate else "inactive",
        endpoint_path=f"/deployments/{deployment_id}/predict",
    )

    db.add(deployment)
    db.commit()
    db.refresh(deployment)

    return deployment_response(deployment)


def deployment_response(deployment: models.ModelDeployment) -> dict:
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
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    deployments = (
        scoped_query(db, models.ModelDeployment, current_user, current_project)
        .order_by(models.ModelDeployment.created_at.desc())
        .all()
    )
    return {"deployments": [deployment_response(item) for item in deployments]}


def get_scoped_deployment(
    db: Session,
    deployment_id: str,
    current_user: models.User,
    current_project: models.Project,
) -> models.ModelDeployment:
    deployment = (
        scoped_query(db, models.ModelDeployment, current_user, current_project)
        .filter(models.ModelDeployment.deployment_id == deployment_id)
        .first()
    )

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found.")

    return deployment


@app.post("/deployments/{deployment_id}/predict")
def predict_with_deployment(
    deployment_id: str,
    request: PredictRequest,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    deployment = get_scoped_deployment(
        db, deployment_id, current_user, current_project
    )

    if deployment.status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Deployment is not active. Current status: {deployment.status}",
        )

    model_record = get_scoped_model(
        db, deployment.model_id, current_user, current_project
    )
    prediction_result = run_prediction_from_model_record(
        model_record=model_record,
        input_data=request.data,
        db=db,
        owner_id=current_user.user_id,
        project_id=current_project.project_id,
    )
    prediction_result.update(
        {
            "deployment_id": deployment.deployment_id,
            "deployment_name": deployment.deployment_name,
            "deployment_version": deployment.version,
            "environment": deployment.environment,
        }
    )
    return prediction_result


@app.patch("/deployments/{deployment_id}/status")
def update_model_deployment_status(
    deployment_id: str,
    request: DeploymentStatusUpdateRequest,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    deployment = get_scoped_deployment(
        db, deployment_id, current_user, current_project
    )

    if request.status == "active":
        (
            scoped_query(db, models.ModelDeployment, current_user, current_project)
            .filter(models.ModelDeployment.deployment_name == deployment.deployment_name)
            .filter(models.ModelDeployment.environment == deployment.environment)
            .filter(models.ModelDeployment.status == "active")
            .update({"status": "inactive"})
        )

    deployment.status = request.status
    db.commit()
    db.refresh(deployment)
    return deployment_response(deployment)


@app.post("/deployments/name/{deployment_name}/versions/{version}/predict")
def predict_with_deployment_version(
    deployment_name: str,
    version: int,
    request: PredictRequest,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    deployment = (
        scoped_query(db, models.ModelDeployment, current_user, current_project)
        .filter(models.ModelDeployment.deployment_name == deployment_name)
        .filter(models.ModelDeployment.version == version)
        .first()
    )

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment version not found.")

    if deployment.status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Deployment is not active. Current status: {deployment.status}",
        )

    model_record = get_scoped_model(
        db, deployment.model_id, current_user, current_project
    )
    prediction_result = run_prediction_from_model_record(
        model_record=model_record,
        input_data=request.data,
        db=db,
        owner_id=current_user.user_id,
        project_id=current_project.project_id,
    )
    prediction_result.update(
        {
            "deployment_id": deployment.deployment_id,
            "deployment_name": deployment.deployment_name,
            "deployment_version": deployment.version,
            "environment": deployment.environment,
        }
    )
    return prediction_result

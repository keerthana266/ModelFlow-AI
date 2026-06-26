import asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from sqlalchemy.orm import Session

from app.ml.predictor import make_prediction
from app.ml.profiler import profile_csv_dataset
from app.ml.trainer import train_automl
from app.ml.transpiler import execute_pipeline_graph
from app.schemas.requests import PredictRequest, TrainRequest, PipelineExecutionRequest
from app.services.storage import (
    create_storage_folders,
    get_dataset_path,
    load_experiment_result,
    save_uploaded_dataset,
)
from app.worker import celery_app
from sqlalchemy.sql import text
from app.db.database import engine

# Import the production database layers
from app.db.database import get_db
from app.db import models

app = FastAPI(
    title="AutoML Forge API",
    description="Production-ready engine for a no-code AutoML platform.",
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
    # 1. Create local storage folders for your datasets
    create_storage_folders()
    
    # 2. Verify PostgreSQL Connection Connectivity
    print("Connecting to PostgreSQL database...")
    try:
        # Establish a temporary connection and run a dummy query
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("DATABASE STATUS: Successfully connected to PostgreSQL!")
    except Exception as e:
        print("DATABASE STATUS: Connection FAILED!")
        print(f"Error Details: {e}")
        print(" Please verify that your PostgreSQL server is running and your .env URL is correct.")
    

@app.get("/")
def root():
    return {
        "message": "AutoML Forge API is running with production PostgreSQL pooling active.",
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
            "**profile": profile,
        }
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error))

@app.post("/experiments/start")
def start_experiment(request: TrainRequest):
    try:
        dataset_path = get_dataset_path(request.dataset_id)

        task = train_automl.delay(
            dataset_path_str=str(dataset_path),
            target_column=request.target_column,
        )

        return {
            "message": "Experiment queued successfully. Polling required.",
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

@app.get("/experiments/{experiment_id}/leaderboard")
def get_experiment_leaderboard(experiment_id: str, db: Session = Depends(get_db)):
    try:
        # First check the centralized PostgreSQL records for the trained result
        record = db.query(models.ExperimentResult).filter(models.ExperimentResult.experiment_id == experiment_id).first()
        
        if record:
            return {
                "experiment_id": record.experiment_id,
                "problem_type": record.problem_type,
                "target_column": record.target_column,
                "main_metric": record.main_metric,
                "best_model_id": record.best_model_id,
                "leaderboard": record.leaderboard,
            }
        
        # Fallback to local storage utilities if the database migration sync is catching up
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
def get_experiment_details(experiment_id: str, db: Session = Depends(get_db)):
    try:
        record = db.query(models.ExperimentResult).filter(models.ExperimentResult.experiment_id == experiment_id).first()
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
            }
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
def save_visual_pipeline(request: PipelineExecutionRequest, db: Session = Depends(get_db)):
    """
    Serializes and commits the present structural configuration of the React Flow frontend 
    canvas directly into a relational PostgreSQL snapshot.
    """
    try:
        new_graph = models.PipelineGraph(
            nodes=[node.model_dump() for node in request.nodes],
            edges=[edge.model_dump() for edge in request.edges]
        )
        db.add(new_graph)
        db.commit()
        db.refresh(new_graph)
        
        return {
            "message": "Visual canvas state committed successfully to PostgreSQL.", 
            "graph_id": str(new_graph.id)
        }
    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database persistence failed to process transaction.")

@app.websocket("/ws/task/{task_id}")
async def task_status_websocket(websocket: WebSocket, task_id: str):
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
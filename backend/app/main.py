from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

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

        # Trigger Celery background task
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

@app.post("/pipeline/execute")
def execute_visual_pipeline(request: PipelineExecutionRequest):
    try:
        # Convert Pydantic objects to standard dictionaries for the transpiler
        nodes_dict = [node.model_dump() for node in request.nodes]
        edges_dict = [edge.model_dump() for edge in request.edges]
        
        result = execute_pipeline_graph(nodes_dict, edges_dict)
        return result
    
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.websocket("/ws/task/{task_id}")
async def task_status_websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    
    try:
        previous_state = None
        previous_message = None
        
        while True:
            # Check the current status of the background task
            task_result = AsyncResult(task_id, app=celery_app)
            current_state = task_result.state
            
            # Extract the meta dictionary we pass via self.update_state() in trainer.py
            meta = task_result.info if isinstance(task_result.info, dict) else {}
            current_message = meta.get("message", "")
            
            # Only send data over the network if something actually changed
            if current_state != previous_state or current_message != previous_message:
                
                payload = {
                    "task_id": task_id,
                    "status": current_state,
                    "message": current_message,
                }
                
                # If finished, attach the final payload or error
                if current_state == "SUCCESS":
                    payload["result"] = task_result.result
                elif current_state == "FAILURE":
                    payload["error"] = str(task_result.result)
                    
                await websocket.send_json(payload)
                
                previous_state = current_state
                previous_message = current_message
            
            # Kill the loop and close the tunnel if the task is finished
            if current_state in ["SUCCESS", "FAILURE"]:
                break
            
            # Pause for half a second before checking again to avoid frying the CPU
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        print(f"Client disconnected from task {task_id} stream.")
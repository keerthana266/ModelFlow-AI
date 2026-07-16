from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    time_budget: int = Field(default=30, ge=10, le=300)
    tuning_mode: Literal["flaml", "optuna"] = "flaml"
    custom_model_ids: List[str] = Field(default_factory=list, max_length=20)
    excluded_columns: List[str] = Field(default_factory=list, max_length=100)
    remove_duplicates: bool = False


class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]


class PipelineNode(BaseModel):
    id: str
    type: str
    data: Dict[str, Any] = {}
    position: Optional[Dict[str, Any]] = None


class PipelineEdge(BaseModel):
    id: str
    source: str
    target: str


class PipelineExecutionRequest(BaseModel):
    nodes: List[PipelineNode]
    edges: List[PipelineEdge]


class DriftCheckRequest(BaseModel):
    model_id: str
    current_dataset_id: Optional[str] = None
    current_data: Optional[List[Dict[str, Any]]] = None
    max_rows: int = Field(default=1000, ge=2, le=10000)


class DeploymentCreateRequest(BaseModel):
    model_id: str
    deployment_name: str = "default-model"
    environment: Literal["development", "staging", "production"] = "production"
    activate: bool = True


class DeploymentStatusUpdateRequest(BaseModel):
    status: Literal["active", "inactive", "archived"]

from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_id: str = Field(..., examples=["dataset_abc123"])
    target_column: str = Field(..., examples=["Survived"])
    time_budget: int = Field(default=30, ge=10, le=300)


class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]


class PipelineNode(BaseModel):
    id: str
    type: str
    data: Dict[str, Any]


class PipelineEdge(BaseModel):
    id: str
    source: str
    target: str


class PipelineExecutionRequest(BaseModel):
    nodes: List[PipelineNode]
    edges: List[PipelineEdge]
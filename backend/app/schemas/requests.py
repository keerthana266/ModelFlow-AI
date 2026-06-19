from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_id: str = Field(..., examples=["dataset_abc123"])
    target_column: str = Field(..., examples=["Survived"])


class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class PipelineNode(BaseModel):
    id: str
    type: str  
    data: Dict[str, Any]  

class PipelineEdge(BaseModel):
    id: str
    source: str  # ID of the node where the wire starts
    target: str  # ID of the node where the wire ends

# The full graph submitted by the frontend
class PipelineExecutionRequest(BaseModel):
    nodes: List[PipelineNode]
    edges: List[PipelineEdge]  
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_id: str = Field(..., examples=["dataset_abc123"])
    target_column: str = Field(..., examples=["Survived"])


class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
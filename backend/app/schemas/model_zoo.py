from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


TaskType = Literal["classification", "regression", "both"]
ModelZooStatus = Literal["draft", "valid", "invalid", "disabled"]
HyperparameterType = Literal["integer", "float", "categorical"]


class HyperparameterDefinition(BaseModel):
    type: HyperparameterType
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None

    @model_validator(mode="after")
    def validate_definition(self):
        if self.type in {"integer", "float"}:
            if self.low is None or self.high is None:
                raise ValueError(
                    f"{self.type} hyperparameters require both low and high."
                )

            if self.low >= self.high:
                raise ValueError("Hyperparameter low must be smaller than high.")

            if self.step is not None and self.step <= 0:
                raise ValueError("Hyperparameter step must be greater than zero.")

            if self.log and self.step is not None:
                raise ValueError(
                    "A hyperparameter cannot use both logarithmic sampling and step."
                )

            if self.type == "integer":
                for field_name, value in (
                    ("low", self.low),
                    ("high", self.high),
                    ("step", self.step),
                ):
                    if value is not None and not float(value).is_integer():
                        raise ValueError(
                            f"Integer hyperparameter {field_name} must be an integer."
                        )

        if self.type == "categorical":
            if not self.choices or len(self.choices) < 2:
                raise ValueError(
                    "Categorical hyperparameters require at least two choices."
                )

            if any(
                value is not None
                for value in (self.low, self.high, self.step)
            ):
                raise ValueError(
                    "Categorical hyperparameters cannot define low, high, or step."
                )

            if self.log:
                raise ValueError(
                    "Categorical hyperparameters cannot use logarithmic sampling."
                )

        return self


class ModelZooCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    description: Optional[str] = Field(default=None, max_length=2000)
    framework: Literal["scikit-learn"] = "scikit-learn"
    task_type: TaskType
    class_path: str = Field(min_length=5, max_length=500)
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    hyperparameter_space: Dict[str, HyperparameterDefinition] = Field(
        default_factory=dict
    )

    @field_validator("name", "class_path")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("Value cannot be empty.")

        return cleaned

    @field_validator("description")
    @classmethod
    def strip_description(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        cleaned = value.strip()
        return cleaned or None


class ModelZooUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=160)
    description: Optional[str] = Field(default=None, max_length=2000)
    framework: Optional[Literal["scikit-learn"]] = None
    task_type: Optional[TaskType] = None
    class_path: Optional[str] = Field(default=None, min_length=5, max_length=500)
    default_parameters: Optional[Dict[str, Any]] = None
    hyperparameter_space: Optional[Dict[str, HyperparameterDefinition]] = None
    status: Optional[Literal["draft", "disabled"]] = None

    @field_validator("name", "class_path")
    @classmethod
    def strip_optional_required_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        cleaned = value.strip()

        if not cleaned:
            raise ValueError("Value cannot be empty.")

        return cleaned

    @field_validator("description")
    @classmethod
    def strip_optional_description(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        cleaned = value.strip()
        return cleaned or None

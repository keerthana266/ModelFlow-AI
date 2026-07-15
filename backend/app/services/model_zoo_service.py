from typing import Any, Dict, Iterable, List

from sqlalchemy.orm import Session

from app.db import models
from app.ml.model_factory import (
    ModelFactoryError,
    validate_model_definition,
)
from app.services.storage import generate_id


class ModelZooServiceError(ValueError):
    pass


class ModelZooNotFoundError(ModelZooServiceError):
    pass


def model_definition_to_dict(record: models.CustomModelDefinition) -> Dict[str, Any]:
    return {
        "model_zoo_id": record.model_zoo_id,
        "name": record.name,
        "description": record.description,
        "framework": record.framework,
        "task_type": record.task_type,
        "class_path": record.class_path,
        "default_parameters": record.default_parameters or {},
        "hyperparameter_space": record.hyperparameter_space or {},
        "status": record.status,
        "validation_error": record.validation_error,
        "validation_details": record.validation_details or {},
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def training_definition_to_dict(
    record: models.CustomModelDefinition,
) -> Dict[str, Any]:
    return {
        "model_zoo_id": record.model_zoo_id,
        "name": record.name,
        "framework": record.framework,
        "task_type": record.task_type,
        "class_path": record.class_path,
        "default_parameters": record.default_parameters or {},
        "hyperparameter_space": record.hyperparameter_space or {},
    }


def get_scoped_model_definition(
    db: Session,
    model_zoo_id: str,
    owner_id: str,
    project_id: str,
) -> models.CustomModelDefinition:
    record = (
        db.query(models.CustomModelDefinition)
        .filter(models.CustomModelDefinition.model_zoo_id == model_zoo_id)
        .filter(models.CustomModelDefinition.owner_id == owner_id)
        .filter(models.CustomModelDefinition.project_id == project_id)
        .first()
    )

    if not record:
        raise ModelZooNotFoundError("Model Zoo definition not found.")

    return record


def list_model_definitions(
    db: Session,
    owner_id: str,
    project_id: str,
) -> List[models.CustomModelDefinition]:
    return (
        db.query(models.CustomModelDefinition)
        .filter(models.CustomModelDefinition.owner_id == owner_id)
        .filter(models.CustomModelDefinition.project_id == project_id)
        .order_by(models.CustomModelDefinition.created_at.desc())
        .all()
    )


def create_model_definition(
    db: Session,
    owner_id: str,
    project_id: str,
    payload,
) -> models.CustomModelDefinition:
    current_count = (
        db.query(models.CustomModelDefinition)
        .filter(models.CustomModelDefinition.owner_id == owner_id)
        .filter(models.CustomModelDefinition.project_id == project_id)
        .count()
    )

    from app.core.config import settings

    if current_count >= settings.MODEL_ZOO_MAX_MODELS_PER_PROJECT:
        raise ModelZooServiceError(
            "This project has reached the configured Model Zoo limit."
        )

    data = payload.model_dump(mode="json")
    record = models.CustomModelDefinition(
        model_zoo_id=generate_id("zoo"),
        owner_id=owner_id,
        project_id=project_id,
        name=data["name"],
        description=data.get("description"),
        framework=data["framework"],
        task_type=data["task_type"],
        class_path=data["class_path"],
        default_parameters=data.get("default_parameters") or {},
        hyperparameter_space=data.get("hyperparameter_space") or {},
        status="draft",
        validation_error=None,
        validation_details={},
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def update_model_definition(
    db: Session,
    record: models.CustomModelDefinition,
    payload,
) -> models.CustomModelDefinition:
    changes = payload.model_dump(
        exclude_unset=True,
        mode="json",
    )

    validation_sensitive_fields = {
        "framework",
        "task_type",
        "class_path",
        "default_parameters",
        "hyperparameter_space",
    }

    changed_validation_field = bool(
        validation_sensitive_fields.intersection(changes)
    )

    for field_name, value in changes.items():
        setattr(record, field_name, value)

    if changed_validation_field:
        record.status = "draft"
        record.validation_error = None
        record.validation_details = {}

    db.commit()
    db.refresh(record)
    return record


def delete_model_definition(
    db: Session,
    record: models.CustomModelDefinition,
) -> None:
    db.delete(record)
    db.commit()


def validate_and_update_model_definition(
    db: Session,
    record: models.CustomModelDefinition,
) -> models.CustomModelDefinition:
    try:
        details = validate_model_definition(record)
        record.status = "valid"
        record.validation_error = None
        record.validation_details = details
    except ModelFactoryError as error:
        record.status = "invalid"
        record.validation_error = str(error)
        record.validation_details = {"valid": False}
    except Exception as error:
        record.status = "invalid"
        record.validation_error = f"Unexpected validation error: {error}"
        record.validation_details = {"valid": False}

    db.commit()
    db.refresh(record)
    return record


def get_training_definitions(
    db: Session,
    model_zoo_ids: Iterable[str] | None,
    owner_id: str,
    project_id: str,
) -> List[Dict[str, Any]]:
    ordered_ids = list(dict.fromkeys(model_zoo_ids or []))

    if not ordered_ids:
        return []

    records = (
        db.query(models.CustomModelDefinition)
        .filter(models.CustomModelDefinition.owner_id == owner_id)
        .filter(models.CustomModelDefinition.project_id == project_id)
        .filter(models.CustomModelDefinition.model_zoo_id.in_(ordered_ids))
        .all()
    )

    records_by_id = {record.model_zoo_id: record for record in records}
    definitions: List[Dict[str, Any]] = []

    for model_zoo_id in ordered_ids:
        record = records_by_id.get(model_zoo_id)

        if not record:
            raise ModelZooNotFoundError(
                f"Model Zoo definition '{model_zoo_id}' was not found "
                "in this project."
            )

        if record.status != "valid":
            raise ModelZooServiceError(
                f"Model Zoo definition '{record.name}' is not valid. "
                "Validate it before starting training."
            )

        definitions.append(training_definition_to_dict(record))

    return definitions

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import models
from app.db.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.project import get_current_project
from app.schemas.model_zoo import (
    ModelZooCreateRequest,
    ModelZooUpdateRequest,
)
from app.services.model_zoo_service import (
    ModelZooNotFoundError,
    ModelZooServiceError,
    create_model_definition,
    delete_model_definition,
    get_scoped_model_definition,
    list_model_definitions,
    model_definition_to_dict,
    update_model_definition,
    validate_and_update_model_definition,
)


router = APIRouter(prefix="/model-zoo", tags=["Model Zoo"])


def _handle_service_error(error: Exception):
    if isinstance(error, ModelZooNotFoundError):
        raise HTTPException(status_code=404, detail=str(error))

    if isinstance(error, ModelZooServiceError):
        raise HTTPException(status_code=400, detail=str(error))

    raise error


@router.get("/capabilities")
def get_model_zoo_capabilities(
    current_user: models.User = Depends(get_current_user),
):
    return {
        "frameworks": ["scikit-learn"],
        "task_types": ["classification", "regression", "both"],
        "hyperparameter_types": ["integer", "float", "categorical"],
        "allowed_modules": settings.model_zoo_allowed_modules_list,
        "security_mode": "allowlisted-class-path",
        "raw_python_uploads_enabled": False,
    }


@router.get("")
def list_project_model_zoo(
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    records = list_model_definitions(
        db,
        owner_id=current_user.user_id,
        project_id=current_project.project_id,
    )

    return {
        "count": len(records),
        "models": [model_definition_to_dict(record) for record in records],
    }


@router.post("", status_code=201)
def create_project_model_zoo_item(
    request: ModelZooCreateRequest,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    try:
        record = create_model_definition(
            db,
            owner_id=current_user.user_id,
            project_id=current_project.project_id,
            payload=request,
        )
        return model_definition_to_dict(record)
    except Exception as error:
        db.rollback()
        _handle_service_error(error)


@router.get("/{model_zoo_id}")
def get_project_model_zoo_item(
    model_zoo_id: str,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    try:
        record = get_scoped_model_definition(
            db,
            model_zoo_id=model_zoo_id,
            owner_id=current_user.user_id,
            project_id=current_project.project_id,
        )
        return model_definition_to_dict(record)
    except Exception as error:
        _handle_service_error(error)


@router.patch("/{model_zoo_id}")
def update_project_model_zoo_item(
    model_zoo_id: str,
    request: ModelZooUpdateRequest,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    try:
        record = get_scoped_model_definition(
            db,
            model_zoo_id=model_zoo_id,
            owner_id=current_user.user_id,
            project_id=current_project.project_id,
        )
        updated = update_model_definition(db, record, request)
        return model_definition_to_dict(updated)
    except Exception as error:
        db.rollback()
        _handle_service_error(error)


@router.delete("/{model_zoo_id}")
def delete_project_model_zoo_item(
    model_zoo_id: str,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    try:
        record = get_scoped_model_definition(
            db,
            model_zoo_id=model_zoo_id,
            owner_id=current_user.user_id,
            project_id=current_project.project_id,
        )
        delete_model_definition(db, record)
        return {
            "message": "Model Zoo definition deleted successfully.",
            "model_zoo_id": model_zoo_id,
        }
    except Exception as error:
        db.rollback()
        _handle_service_error(error)


@router.post("/{model_zoo_id}/validate")
def validate_project_model_zoo_item(
    model_zoo_id: str,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    try:
        record = get_scoped_model_definition(
            db,
            model_zoo_id=model_zoo_id,
            owner_id=current_user.user_id,
            project_id=current_project.project_id,
        )
        validated = validate_and_update_model_definition(db, record)
        return model_definition_to_dict(validated)
    except Exception as error:
        db.rollback()
        _handle_service_error(error)

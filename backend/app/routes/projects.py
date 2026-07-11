from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import models
from app.db.database import get_db
from app.dependencies.auth import get_current_user
from app.schemas.project import (
    ProjectCreateRequest,
    ProjectResponse,
    ProjectUpdateRequest,
)
from app.services.storage import generate_id


router = APIRouter(prefix="/projects", tags=["Projects"])


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(
    request: ProjectCreateRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = models.Project(
        project_id=generate_id("project"),
        owner_id=current_user.user_id,
        name=request.name.strip(),
        description=request.description,
    )

    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("")
def list_projects(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    projects = (
        db.query(models.Project)
        .filter(models.Project.owner_id == current_user.user_id)
        .order_by(models.Project.updated_at.desc())
        .all()
    )

    return {
        "count": len(projects),
        "projects": [ProjectResponse.model_validate(project) for project in projects],
    }


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = (
        db.query(models.Project)
        .filter(models.Project.project_id == project_id)
        .filter(models.Project.owner_id == current_user.user_id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str,
    request: ProjectUpdateRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = (
        db.query(models.Project)
        .filter(models.Project.project_id == project_id)
        .filter(models.Project.owner_id == current_user.user_id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    if request.name is not None:
        project.name = request.name.strip()

    if request.description is not None:
        project.description = request.description

    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}")
def delete_project(
    project_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = (
        db.query(models.Project)
        .filter(models.Project.project_id == project_id)
        .filter(models.Project.owner_id == current_user.user_id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    project_count = (
        db.query(models.Project)
        .filter(models.Project.owner_id == current_user.user_id)
        .count()
    )

    if project_count <= 1:
        raise HTTPException(
            status_code=409,
            detail="Create another project before deleting your only project.",
        )

    resource_models = (
        models.DatasetRecord,
        models.ExperimentResult,
        models.TrainedModel,
        models.PredictionLog,
        models.PipelineGraph,
        models.DriftReport,
        models.ModelDeployment,
    )

    has_resources = any(
        db.query(resource_model)
        .filter(resource_model.project_id == project_id)
        .first()
        is not None
        for resource_model in resource_models
    )

    if has_resources:
        raise HTTPException(
            status_code=409,
            detail="This project contains resources and cannot be deleted yet.",
        )

    db.delete(project)
    db.commit()
    return {"message": "Project deleted successfully.", "project_id": project_id}

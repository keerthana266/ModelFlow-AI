from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import models
from app.db.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.project import get_current_project
from app.services.dataset_readiness import analyze_dataset_readiness


router = APIRouter(tags=["Dataset Readiness"])


@router.get("/datasets/{dataset_id}/readiness")
def get_dataset_readiness(
    dataset_id: str,
    target_column: Annotated[str | None, Query()] = None,
    current_user: models.User = Depends(get_current_user),
    current_project: models.Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = (
        db.query(models.DatasetRecord)
        .filter(models.DatasetRecord.dataset_id == dataset_id)
        .filter(models.DatasetRecord.owner_id == current_user.user_id)
        .filter(models.DatasetRecord.project_id == current_project.project_id)
        .first()
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found in this project.",
        )

    try:
        return {
            "dataset_id": dataset.dataset_id,
            "original_filename": dataset.original_filename,
            "target_column": target_column,
            "readiness": analyze_dataset_readiness(
                Path(dataset.file_path),
                target_column=target_column,
            ),
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

from typing import Annotated

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db import models
from app.db.database import get_db
from app.dependencies.auth import get_current_user


def get_current_project(
    current_user: models.User = Depends(get_current_user),
    project_id: Annotated[
        str | None,
        Header(alias="X-Project-ID"),
    ] = None,
    db: Session = Depends(get_db),
) -> models.Project:
    if not project_id:
        owned_projects = (
            db.query(models.Project)
            .filter(models.Project.owner_id == current_user.user_id)
            .order_by(models.Project.created_at.asc())
            .all()
        )

        if len(owned_projects) == 1:
            return owned_projects[0]

        raise HTTPException(
            status_code=400,
            detail="Select a project and send its ID in the X-Project-ID header.",
        )

    project = (
        db.query(models.Project)
        .filter(models.Project.project_id == project_id)
        .filter(models.Project.owner_id == current_user.user_id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    return project

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import models
from app.db.database import get_db
from app.dependencies.auth import get_current_user
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth_service import (
    authenticate_user,
    create_login_token,
    register_user,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user, project, token = register_user(
            db,
            name=request.name,
            email=str(request.email),
            password=request.password,
        )

        return TokenResponse(
            access_token=token,
            user=UserResponse.model_validate(user),
            default_project_id=project.project_id,
        )
    except ValueError as error:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(error))


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(
        db,
        email=str(request.email),
        password=request.password,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    default_project = (
        db.query(models.Project)
        .filter(models.Project.owner_id == user.user_id)
        .order_by(models.Project.created_at.asc())
        .first()
    )

    return TokenResponse(
        access_token=create_login_token(user),
        user=UserResponse.model_validate(user),
        default_project_id=(
            default_project.project_id if default_project else None
        ),
    )


@router.get("/me", response_model=UserResponse)
def read_me(current_user: models.User = Depends(get_current_user)):
    return current_user

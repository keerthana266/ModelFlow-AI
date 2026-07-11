from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.db import models
from app.services.storage import generate_id


def normalize_email(email: str) -> str:
    return email.strip().lower()


def register_user(
    db: Session,
    *,
    name: str,
    email: str,
    password: str,
) -> tuple[models.User, models.Project, str]:
    normalized_email = normalize_email(email)

    existing_user = (
        db.query(models.User)
        .filter(models.User.email == normalized_email)
        .first()
    )

    if existing_user:
        raise ValueError("An account with this email already exists.")

    user = models.User(
        user_id=generate_id("user"),
        name=name.strip(),
        email=normalized_email,
        password_hash=hash_password(password),
        is_active=True,
    )

    project = models.Project(
        project_id=generate_id("project"),
        owner_id=user.user_id,
        name="My AutoML Project",
        description="Default workspace created during registration.",
    )

    db.add(user)
    db.add(project)
    db.commit()
    db.refresh(user)
    db.refresh(project)

    token = create_access_token(
        subject=user.user_id,
        extra={"email": user.email},
    )

    return user, project, token


def authenticate_user(
    db: Session,
    *,
    email: str,
    password: str,
) -> models.User | None:
    normalized_email = normalize_email(email)
    user = (
        db.query(models.User)
        .filter(models.User.email == normalized_email)
        .first()
    )

    if not user or not user.is_active:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user


def create_login_token(user: models.User) -> str:
    return create_access_token(
        subject=user.user_id,
        extra={"email": user.email},
    )

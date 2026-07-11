from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db import models
from app.db.database import get_db


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    db: Session = Depends(get_db),
) -> models.User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise credentials_error

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = str(payload["sub"])
    except (ValueError, KeyError):
        raise credentials_error

    user = (
        db.query(models.User)
        .filter(models.User.user_id == user_id)
        .first()
    )

    if not user or not user.is_active:
        raise credentials_error

    return user

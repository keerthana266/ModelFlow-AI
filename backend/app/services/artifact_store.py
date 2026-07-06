# app/services/artifact_store.py

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings


def artifact_store_enabled() -> bool:
    return settings.ARTIFACT_STORE_BACKEND.lower() == "minio"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY_ID,
        aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
        region_name=settings.S3_REGION_NAME,
    )


def ensure_bucket_exists() -> None:
    client = get_s3_client()
    bucket_name = settings.S3_BUCKET_NAME

    try:
        client.head_bucket(Bucket=bucket_name)

    except ClientError:
        client.create_bucket(Bucket=bucket_name)


def build_artifact_response(
    object_key: str,
    content_type: str,
    size_bytes: int | None = None,
) -> Dict[str, Any]:
    return {
        "backend": "minio",
        "bucket": settings.S3_BUCKET_NAME,
        "object_key": object_key,
        "s3_uri": f"s3://{settings.S3_BUCKET_NAME}/{object_key}",
        "endpoint_url": settings.S3_ENDPOINT_URL,
        "content_type": content_type,
        "size_bytes": size_bytes,
    }


def upload_file_artifact(
    local_path: str | Path,
    object_key: str,
    content_type: str = "application/octet-stream",
) -> Dict[str, Any]:
    if not artifact_store_enabled():
        return {
            "enabled": False,
            "backend": settings.ARTIFACT_STORE_BACKEND,
            "message": "Artifact store is not enabled.",
        }

    local_path = Path(local_path)

    if not local_path.exists():
        return {
            "enabled": True,
            "backend": "minio",
            "status": "failed",
            "error": f"Local file not found: {local_path}",
        }

    try:
        ensure_bucket_exists()

        client = get_s3_client()

        client.upload_file(
            Filename=str(local_path),
            Bucket=settings.S3_BUCKET_NAME,
            Key=object_key,
            ExtraArgs={"ContentType": content_type},
        )

        return {
            "enabled": True,
            "status": "uploaded",
            **build_artifact_response(
                object_key=object_key,
                content_type=content_type,
                size_bytes=local_path.stat().st_size,
            ),
        }

    except Exception as error:
        return {
            "enabled": True,
            "backend": "minio",
            "status": "failed",
            "error": str(error),
        }


def upload_json_artifact(
    payload: Dict[str, Any],
    object_key: str,
) -> Dict[str, Any]:
    if not artifact_store_enabled():
        return {
            "enabled": False,
            "backend": settings.ARTIFACT_STORE_BACKEND,
            "message": "Artifact store is not enabled.",
        }

    try:
        ensure_bucket_exists()

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            json.dump(payload, temp_file, indent=2, default=str)
            temp_path = Path(temp_file.name)

        try:
            return upload_file_artifact(
                local_path=temp_path,
                object_key=object_key,
                content_type="application/json",
            )

        finally:
            temp_path.unlink(missing_ok=True)

    except Exception as error:
        return {
            "enabled": True,
            "backend": "minio",
            "status": "failed",
            "error": str(error),
        }


def upload_model_artifact(
    local_model_path: str | Path,
    experiment_id: str,
    model_id: str,
) -> Dict[str, Any]:
    object_key = f"experiments/{experiment_id}/models/{model_id}.joblib"

    return upload_file_artifact(
        local_path=local_model_path,
        object_key=object_key,
        content_type="application/octet-stream",
    )


def upload_experiment_artifact(
    result: Dict[str, Any],
    experiment_id: str,
) -> Dict[str, Any]:
    object_key = f"experiments/{experiment_id}/result.json"

    return upload_json_artifact(
        payload=result,
        object_key=object_key,
    )


def get_artifact_store_status() -> Dict[str, Any]:
    status = {
        "backend": settings.ARTIFACT_STORE_BACKEND,
        "enabled": artifact_store_enabled(),
        "bucket": settings.S3_BUCKET_NAME,
        "endpoint_url": settings.S3_ENDPOINT_URL,
    }

    if not artifact_store_enabled():
        status["status"] = "disabled"
        return status

    try:
        ensure_bucket_exists()

        client = get_s3_client()
        client.head_bucket(Bucket=settings.S3_BUCKET_NAME)

        status["status"] = "online"
        status["message"] = "MinIO/S3 artifact store is reachable."

        return status

    except Exception as error:
        status["status"] = "offline"
        status["error"] = str(error)

        return status
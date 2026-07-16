from __future__ import annotations

from datetime import datetime

from app.db import models
from app.db.database import SessionLocal
from app.services.docker_deployment_service import (
    DockerDeploymentError,
    remove_container,
    start_existing_container,
    start_new_model_container,
    stop_container,
)
from app.worker import celery_app


@celery_app.task(bind=True, name="deploy_model_container_task")
def deploy_model_container_task(
    self,
    deployment_id: str,
    recreate: bool = False,
) -> dict:
    db = SessionLocal()
    deployment = None

    try:
        deployment = (
            db.query(models.ModelDeployment)
            .filter(models.ModelDeployment.deployment_id == deployment_id)
            .first()
        )
        if not deployment:
            raise ValueError(f"Deployment '{deployment_id}' was not found.")

        model = (
            db.query(models.TrainedModel)
            .filter(models.TrainedModel.model_id == deployment.model_id)
            .filter(models.TrainedModel.owner_id == deployment.owner_id)
            .filter(models.TrainedModel.project_id == deployment.project_id)
            .first()
        )
        if not model:
            raise ValueError("The trained model for this deployment was not found.")

        deployment.task_id = self.request.id
        deployment.status = "starting" if deployment.container_id else "building"
        deployment.error_message = None
        db.commit()

        self.update_state(
            state="PROGRESS",
            meta={"message": "Preparing isolated model container..."},
        )

        runtime_info = None
        if deployment.container_id and not recreate and deployment.health_url:
            try:
                runtime_info = start_existing_container(
                    deployment.container_id,
                    deployment.health_url,
                )
            except DockerDeploymentError:
                # The previous container may have been manually deleted.
                runtime_info = None

        if runtime_info is None:
            if deployment.container_id:
                remove_container(deployment.container_id)

            self.update_state(
                state="PROGRESS",
                meta={"message": "Starting Docker model runtime..."},
            )
            runtime_info = start_new_model_container(
                deployment_id=deployment.deployment_id,
                deployment_name=deployment.deployment_name,
                version=deployment.version,
                owner_id=deployment.owner_id,
                project_id=deployment.project_id,
                model_id=deployment.model_id,
                model_path=model.model_path,
            )

            deployment.container_id = runtime_info["container_id"]
            deployment.container_name = runtime_info["container_name"]
            deployment.runtime_image = runtime_info["runtime_image"]
            deployment.host = runtime_info["host"]
            deployment.host_port = runtime_info["host_port"]
            deployment.container_port = runtime_info["container_port"]
            deployment.runtime_endpoint_url = runtime_info["runtime_endpoint_url"]
            deployment.health_url = runtime_info["health_url"]

        if deployment.activate_on_ready:
            previous_active = (
                db.query(models.ModelDeployment)
                .filter(models.ModelDeployment.owner_id == deployment.owner_id)
                .filter(models.ModelDeployment.project_id == deployment.project_id)
                .filter(
                    models.ModelDeployment.deployment_name
                    == deployment.deployment_name
                )
                .filter(models.ModelDeployment.environment == deployment.environment)
                .filter(models.ModelDeployment.status == "active")
                .filter(
                    models.ModelDeployment.deployment_id
                    != deployment.deployment_id
                )
                .all()
            )
            for previous in previous_active:
                if previous.container_id:
                    try:
                        stop_container(previous.container_id)
                    except DockerDeploymentError:
                        pass
                previous.status = "inactive"
                previous.stopped_at = datetime.utcnow()

            deployment.status = "active"
        else:
            if deployment.container_id:
                try:
                    stop_container(deployment.container_id)
                except DockerDeploymentError:
                    pass
            deployment.status = "inactive"
            deployment.stopped_at = datetime.utcnow()

        deployment.runtime_logs = runtime_info.get("runtime_logs")
        deployment.started_at = datetime.utcnow()
        if deployment.status == "active":
            deployment.stopped_at = None
        deployment.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(deployment)

        return {
            "deployment_id": deployment.deployment_id,
            "status": deployment.status,
            "container_id": deployment.container_id,
            "container_name": deployment.container_name,
            "host_port": deployment.host_port,
            "runtime_endpoint_url": deployment.runtime_endpoint_url,
            "health_url": deployment.health_url,
        }

    except Exception as error:
        db.rollback()
        failed_deployment = (
            db.query(models.ModelDeployment)
            .filter(models.ModelDeployment.deployment_id == deployment_id)
            .first()
        )
        if failed_deployment:
            failed_deployment.status = "failed"
            failed_deployment.error_message = str(error)
            failed_deployment.updated_at = datetime.utcnow()
            db.commit()
        raise
    finally:
        db.close()

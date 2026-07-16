from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import docker
import httpx
from docker.errors import DockerException, ImageNotFound, NotFound

from app.core.config import settings


class DockerDeploymentError(RuntimeError):
    """Raised when a Docker model-deployment operation fails."""


def get_docker_client():
    """
    Create and validate a connection to the local Docker Engine.
    """
    try:
        client = docker.from_env()
        client.ping()
        return client
    except DockerException as error:
        raise DockerDeploymentError(
            "Docker Engine is unavailable. Start Docker Desktop and ensure "
            "the Celery worker can access the Docker daemon."
        ) from error


def sanitize_container_name(value: str) -> str:
    """
    Convert a user-provided deployment name into a valid Docker container name.
    """
    cleaned = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "-",
        str(value or ""),
    ).strip("-.")

    return cleaned[:120] or "automl-model"


def ensure_runtime_image(client) -> None:
    """
    Ensure the reusable AutoML Forge model-runtime image exists locally.
    """
    try:
        client.images.get(settings.DOCKER_RUNTIME_IMAGE)
    except ImageNotFound as error:
        raise DockerDeploymentError(
            f"Docker runtime image '{settings.DOCKER_RUNTIME_IMAGE}' "
            "was not found. Build backend/model_runtime first."
        ) from error
    except DockerException as error:
        raise DockerDeploymentError(
            f"Could not inspect Docker runtime image: {error}"
        ) from error


def get_container(identifier: str | None):
    """
    Find a Docker container using its ID or name.
    """
    if not identifier:
        return None

    client = get_docker_client()

    try:
        return client.containers.get(identifier)
    except NotFound:
        return None
    except DockerException as error:
        raise DockerDeploymentError(
            f"Could not inspect Docker container: {error}"
        ) from error


def remove_container(identifier: str | None) -> None:
    """
    Forcefully remove a container when it exists.
    """
    container = get_container(identifier)

    if not container:
        return

    try:
        container.remove(force=True)
    except NotFound:
        return
    except DockerException as error:
        raise DockerDeploymentError(
            f"Could not remove Docker container: {error}"
        ) from error


def container_logs(identifier: str | None, tail: int = 250) -> str:
    """
    Return the latest Docker logs for a deployment container.
    """
    container = get_container(identifier)

    if not container:
        return "Container does not exist."

    try:
        value = container.logs(
            tail=tail,
            timestamps=True,
        )

        return value.decode(
            "utf-8",
            errors="replace",
        )
    except DockerException as error:
        raise DockerDeploymentError(
            f"Could not read Docker container logs: {error}"
        ) from error


def wait_for_health(
    health_url: str,
    timeout_seconds: int | None = None,
) -> dict:
    """
    Poll the model-runtime health endpoint until it becomes healthy.
    """
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.DOCKER_HEALTH_TIMEOUT_SECONDS
    )

    deadline = time.monotonic() + timeout
    last_error = "Runtime did not answer."

    while time.monotonic() < deadline:
        try:
            response = httpx.get(
                health_url,
                timeout=2.0,
            )

            if response.status_code == 200:
                payload = response.json()

                if payload.get("status") == "healthy":
                    return payload

                last_error = f"Unexpected health payload: {payload}"
            else:
                last_error = (
                    f"Health endpoint returned HTTP "
                    f"{response.status_code}."
                )

        except Exception as error:
            # The container may still be starting.
            last_error = str(error)

        time.sleep(1)

    raise DockerDeploymentError(
        f"Model container failed its health check: {last_error}"
    )


def get_published_host_port(
    container,
    container_port: int,
    timeout_seconds: int = 10,
) -> int:
    """
    Read the random host port assigned by Docker.

    Docker receives:
        ports={"8000/tcp": None}

    It then selects a currently available host port automatically.
    """
    port_key = f"{container_port}/tcp"
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            container.reload()

            bindings = (
                container.attrs
                .get("NetworkSettings", {})
                .get("Ports", {})
                .get(port_key)
            )

            if bindings:
                host_port = bindings[0].get("HostPort")

                if host_port:
                    return int(host_port)

        except DockerException as error:
            raise DockerDeploymentError(
                f"Could not inspect Docker port mapping: {error}"
            ) from error

        time.sleep(0.25)

    raise DockerDeploymentError(
        f"Docker did not publish container port {container_port}."
    )


def remove_stale_container(client, container_name: str) -> None:
    """
    Remove an old container left behind by a failed or retried deployment.
    """
    try:
        stale_container = client.containers.get(container_name)
        stale_container.remove(force=True)
    except NotFound:
        pass
    except DockerException as error:
        raise DockerDeploymentError(
            f"Could not remove stale deployment container: {error}"
        ) from error


def start_new_model_container(
    *,
    deployment_id: str,
    deployment_name: str,
    version: int,
    owner_id: str,
    project_id: str,
    model_id: str,
    model_path: str,
) -> dict[str, Any]:
    """
    Start a new isolated Docker container for one trained model.

    Docker automatically chooses an available host port. This prevents
    collisions such as multiple deployments trying to use port 8101.
    """
    client = get_docker_client()
    ensure_runtime_image(client)

    resolved_model_path = Path(model_path).expanduser().resolve()

    if not resolved_model_path.exists():
        raise DockerDeploymentError(
            f"Model artifact does not exist: {resolved_model_path}"
        )

    if not resolved_model_path.is_file():
        raise DockerDeploymentError(
            f"Model artifact path is not a file: {resolved_model_path}"
        )

    container_name = sanitize_container_name(
        f"automl-{deployment_name}-v{version}-{deployment_id[-8:]}"
    )

    remove_stale_container(
        client=client,
        container_name=container_name,
    )

    container_port = settings.DOCKER_CONTAINER_PORT
    container = None

    try:
        container = client.containers.run(
            image=settings.DOCKER_RUNTIME_IMAGE,
            name=container_name,
            detach=True,
            environment={
                "MODEL_PATH": "/models/model.joblib",
            },
            volumes={
                str(resolved_model_path): {
                    "bind": "/models/model.joblib",
                    "mode": "ro",
                }
            },

            # Let Docker select an available host port automatically.
            ports={
                f"{container_port}/tcp": None,
            },

            labels={
                "automl-forge.managed": "true",
                "automl-forge.deployment-id": deployment_id,
                "automl-forge.model-id": model_id,
                "automl-forge.owner-id": owner_id,
                "automl-forge.project-id": project_id,
            },
            restart_policy={
                "Name": "unless-stopped",
            },
            mem_limit=settings.DOCKER_MEMORY_LIMIT,
            nano_cpus=settings.DOCKER_NANO_CPUS,
            pids_limit=settings.DOCKER_PIDS_LIMIT,
            read_only=True,
            tmpfs={
                "/tmp": "rw,noexec,nosuid,size=67108864",
            },
            cap_drop=[
                "ALL",
            ],
            security_opt=[
                "no-new-privileges:true",
            ],
        )

        host_port = get_published_host_port(
            container=container,
            container_port=container_port,
        )

        runtime_base_url = (
            f"http://{settings.DOCKER_DEPLOYMENT_HOST}:{host_port}"
        )

        health_url = f"{runtime_base_url}/health"

        health = wait_for_health(health_url)

        container.reload()

        return {
            "container_id": container.id,
            "container_name": container.name,
            "runtime_image": settings.DOCKER_RUNTIME_IMAGE,
            "host": settings.DOCKER_DEPLOYMENT_HOST,
            "host_port": host_port,
            "container_port": container_port,
            "runtime_endpoint_url": f"{runtime_base_url}/predict",
            "health_url": health_url,
            "health": health,
            "runtime_logs": container_logs(container.id),
        }

    except Exception as error:
        logs = ""

        if container is not None:
            try:
                logs = container_logs(container.id)
            except Exception:
                logs = ""

            try:
                container.remove(force=True)
            except Exception:
                pass
        else:
            # Docker may create the container but fail while starting it.
            try:
                failed_container = client.containers.get(container_name)
                logs = failed_container.logs(
                    tail=250,
                    timestamps=True,
                ).decode("utf-8", errors="replace")
                failed_container.remove(force=True)
            except Exception:
                pass

        if isinstance(error, DockerDeploymentError):
            message = str(error)
        else:
            message = f"Docker could not start the model container: {error}"

        if logs:
            message = f"{message}\n\nContainer logs:\n{logs[-4000:]}"

        raise DockerDeploymentError(message) from error


def start_existing_container(
    identifier: str,
    health_url: str,
) -> dict:
    """
    Start an existing stopped deployment container.
    """
    container = get_container(identifier)

    if not container:
        raise DockerDeploymentError(
            "The deployment container no longer exists."
        )

    try:
        container.reload()

        if container.status != "running":
            container.start()

        health = wait_for_health(health_url)

        container.reload()

        return {
            "status": container.status,
            "health": health,
            "runtime_logs": container_logs(container.id),
        }

    except DockerDeploymentError:
        raise
    except DockerException as error:
        raise DockerDeploymentError(
            f"Could not start deployment container: {error}"
        ) from error


def stop_container(identifier: str) -> str:
    """
    Stop a running deployment container.
    """
    container = get_container(identifier)

    if not container:
        raise DockerDeploymentError(
            "The deployment container no longer exists."
        )

    try:
        container.stop(timeout=10)
        container.reload()
        return container.status

    except DockerException as error:
        raise DockerDeploymentError(
            f"Could not stop deployment container: {error}"
        ) from error


def restart_container(
    identifier: str,
    health_url: str,
) -> dict:
    """
    Restart a deployment container and verify that it becomes healthy.
    """
    container = get_container(identifier)

    if not container:
        raise DockerDeploymentError(
            "The deployment container no longer exists."
        )

    try:
        container.restart(timeout=10)

        health = wait_for_health(health_url)

        container.reload()

        return {
            "status": container.status,
            "health": health,
            "runtime_logs": container_logs(container.id),
        }

    except DockerDeploymentError:
        raise
    except DockerException as error:
        raise DockerDeploymentError(
            f"Could not restart deployment container: {error}"
        ) from error


def inspect_container(identifier: str | None) -> dict:
    """
    Return Docker metadata for a deployment container.
    """
    container = get_container(identifier)

    if not container:
        return {
            "exists": False,
            "status": "missing",
        }

    try:
        container.reload()

        return {
            "exists": True,
            "container_id": container.id,
            "container_name": container.name,
            "status": container.status,
            "image": container.image.tags,
            "created": container.attrs.get("Created"),
            "state": container.attrs.get("State", {}),
            "ports": (
                container.attrs
                .get("NetworkSettings", {})
                .get("Ports", {})
            ),
        }

    except DockerException as error:
        raise DockerDeploymentError(
            f"Could not inspect deployment container: {error}"
        ) from error
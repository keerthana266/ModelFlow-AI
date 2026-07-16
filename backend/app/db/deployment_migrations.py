from sqlalchemy import inspect, text

from app.db.database import engine


DEPLOYMENT_COLUMNS = {
    "activate_on_ready": "BOOLEAN NOT NULL DEFAULT TRUE",
    "task_id": "VARCHAR",
    "runtime_image": "VARCHAR",
    "container_id": "VARCHAR",
    "container_name": "VARCHAR",
    "host": "VARCHAR DEFAULT '127.0.0.1'",
    "host_port": "INTEGER",
    "container_port": "INTEGER DEFAULT 8000",
    "runtime_endpoint_url": "TEXT",
    "health_url": "TEXT",
    "error_message": "TEXT",
    "runtime_logs": "TEXT",
    "updated_at": "TIMESTAMP",
    "started_at": "TIMESTAMP",
    "stopped_at": "TIMESTAMP",
}


def ensure_docker_deployment_columns() -> None:
    """Add Docker deployment columns to an existing PostgreSQL table.

    ``Base.metadata.create_all`` creates missing tables but does not add new
    columns to tables that already exist. This lightweight migration keeps the
    current project working without requiring the user to generate an Alembic
    revision immediately.
    """
    inspector = inspect(engine)

    if "model_deployments" not in inspector.get_table_names():
        return

    existing = {
        column["name"] for column in inspector.get_columns("model_deployments")
    }

    with engine.begin() as connection:
        for column_name, sql_type in DEPLOYMENT_COLUMNS.items():
            if column_name in existing:
                continue

            connection.execute(
                text(
                    f'ALTER TABLE model_deployments '
                    f'ADD COLUMN "{column_name}" {sql_type}'
                )
            )

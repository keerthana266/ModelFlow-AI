

from sqlalchemy import inspect, text

from app.db.database import engine


SCOPED_TABLES = (
    "datasets",
    "experiment_results",
    "trained_models",
    "prediction_logs",
    "pipeline_graphs",
    "drift_reports",
    "model_deployments",
)


def ensure_project_scope_columns() -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as connection:
        for table_name in SCOPED_TABLES:
            if table_name not in existing_tables:
                continue

            existing_columns = {
                column["name"]
                for column in inspector.get_columns(table_name)
            }

            if "owner_id" not in existing_columns:
                connection.execute(
                    text(f'ALTER TABLE "{table_name}" ADD COLUMN owner_id VARCHAR')
                )

            if "project_id" not in existing_columns:
                connection.execute(
                    text(f'ALTER TABLE "{table_name}" ADD COLUMN project_id VARCHAR')
                )

            connection.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS ix_{table_name}_owner_id '
                    f'ON "{table_name}" (owner_id)'
                )
            )
            connection.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS ix_{table_name}_project_id '
                    f'ON "{table_name}" (project_id)'
                )
            )

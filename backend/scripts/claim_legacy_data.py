

import argparse

from app.db import models
from app.db.database import SessionLocal


RESOURCE_MODELS = (
    models.DatasetRecord,
    models.ExperimentResult,
    models.TrainedModel,
    models.PredictionLog,
    models.PipelineGraph,
    models.DriftReport,
    models.ModelDeployment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--project-id", default=None)
    args = parser.parse_args()

    db = SessionLocal()

    try:
        user = (
            db.query(models.User)
            .filter(models.User.email == args.email.strip().lower())
            .first()
        )

        if not user:
            raise SystemExit("User not found. Register the account first.")

        project_query = db.query(models.Project).filter(
            models.Project.owner_id == user.user_id
        )

        if args.project_id:
            project_query = project_query.filter(
                models.Project.project_id == args.project_id
            )

        project = project_query.order_by(models.Project.created_at.asc()).first()

        if not project:
            raise SystemExit("No matching project was found for this user.")

        claimed = {}

        for resource_model in RESOURCE_MODELS:
            count = (
                db.query(resource_model)
                .filter(resource_model.project_id.is_(None))
                .update(
                    {
                        "owner_id": user.user_id,
                        "project_id": project.project_id,
                    },
                    synchronize_session=False,
                )
            )
            claimed[resource_model.__tablename__] = count

        db.commit()

        print(f"Claimed legacy records for project: {project.project_id}")
        for table_name, count in claimed.items():
            print(f"  {table_name}: {count}")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()

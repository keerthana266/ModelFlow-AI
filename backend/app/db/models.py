from datetime import datetime
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class Project(Base):
    __tablename__ = "projects"

    project_id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, index=True, nullable=False)
    name = Column(String(160), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class TaskRun(Base):
    __tablename__ = "task_runs"

    task_id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, index=True, nullable=False)
    project_id = Column(String, index=True, nullable=False)
    task_type = Column(String(60), nullable=False, default="training")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DatasetRecord(Base):
    __tablename__ = "datasets"

    dataset_id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, index=True, nullable=True)
    project_id = Column(String, index=True, nullable=True)

    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)

    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)
    column_names = Column(JSON, nullable=False)
    dtypes = Column(JSON, nullable=True)
    missing_values = Column(JSON, nullable=True)
    unique_values = Column(JSON, nullable=True)
    duplicate_rows = Column(Integer, nullable=True)

    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())


class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    experiment_id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, index=True, nullable=True)
    project_id = Column(String, index=True, nullable=True)

    problem_type = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    rows_used = Column(Integer)
    best_model_id = Column(String, nullable=False)
    main_metric = Column(String)
    best_score = Column(Float)
    feature_importance = Column(JSON, nullable=True)
    leaderboard = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CustomModelDefinition(Base):
    __tablename__ = "custom_model_definitions"

    model_zoo_id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, index=True, nullable=False)
    project_id = Column(String, index=True, nullable=False)

    name = Column(String(160), nullable=False)
    description = Column(Text, nullable=True)
    framework = Column(String(60), nullable=False, default="scikit-learn")
    task_type = Column(String(40), nullable=False)
    class_path = Column(String(500), nullable=False)

    default_parameters = Column(JSON, nullable=False, default=dict)
    hyperparameter_space = Column(JSON, nullable=False, default=dict)

    status = Column(String(30), nullable=False, default="draft")
    validation_error = Column(Text, nullable=True)
    validation_details = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class TrainedModel(Base):
    __tablename__ = "trained_models"

    model_id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, index=True, nullable=True)
    project_id = Column(String, index=True, nullable=True)

    experiment_id = Column(String, index=True, nullable=False)
    model_name = Column(String, nullable=False)
    problem_type = Column(String, nullable=False)
    target_column = Column(String, nullable=False)

    main_metric = Column(String)
    score = Column(Float)
    metrics = Column(JSON, nullable=False)

    feature_columns = Column(JSON, nullable=False)
    numerical_columns = Column(JSON, nullable=True)
    categorical_columns = Column(JSON, nullable=True)

    model_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    prediction_id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, index=True, nullable=True)
    project_id = Column(String, index=True, nullable=True)

    model_id = Column(String, index=True, nullable=False)
    input_data = Column(JSON, nullable=False)
    prediction = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PipelineGraph(Base):
    __tablename__ = "pipeline_graphs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    owner_id = Column(String, index=True, nullable=True)
    project_id = Column(String, index=True, nullable=True)

    nodes = Column(JSON, nullable=False)
    edges = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DriftReport(Base):
    __tablename__ = "drift_reports"

    report_id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, index=True, nullable=True)
    project_id = Column(String, index=True, nullable=True)

    model_id = Column(String, index=True, nullable=False)
    experiment_id = Column(String, index=True, nullable=True)
    current_dataset_id = Column(String, index=True, nullable=True)

    drift_detected = Column(Boolean, default=False)
    drift_score = Column(Float, nullable=True)
    drifted_columns = Column(Integer, default=0)
    total_columns = Column(Integer, default=0)

    report_path = Column(Text, nullable=False)
    summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    deployment_id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, index=True, nullable=True)
    project_id = Column(String, index=True, nullable=True)

    deployment_name = Column(String, index=True, nullable=False)
    model_id = Column(String, index=True, nullable=False)
    experiment_id = Column(String, index=True, nullable=True)

    version = Column(Integer, nullable=False)
    environment = Column(String, default="production")
    status = Column(String, default="pending", index=True)
    activate_on_ready = Column(Boolean, default=True, nullable=False)

    # Stable authenticated proxy endpoint exposed by the main API.
    endpoint_path = Column(Text, nullable=False)

    # Celery orchestration metadata.
    task_id = Column(String, index=True, nullable=True)

    # Docker runtime metadata.
    runtime_image = Column(String, nullable=True)
    container_id = Column(String, nullable=True)
    container_name = Column(String, nullable=True)
    host = Column(String, default="127.0.0.1")
    host_port = Column(Integer, nullable=True)
    container_port = Column(Integer, default=8000)
    runtime_endpoint_url = Column(Text, nullable=True)
    health_url = Column(Text, nullable=True)

    error_message = Column(Text, nullable=True)
    runtime_logs = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)

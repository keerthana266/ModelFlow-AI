# app/db/models.py

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
from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


class DatasetRecord(Base):
    __tablename__ = "datasets"

    dataset_id = Column(String, primary_key=True, index=True)
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
    problem_type = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    rows_used = Column(Integer)
    best_model_id = Column(String, nullable=False)
    main_metric = Column(String)
    best_score = Column(Float)
    feature_importance = Column(JSON, nullable=True)
    leaderboard = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TrainedModel(Base):
    __tablename__ = "trained_models"

    model_id = Column(String, primary_key=True, index=True)
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
    model_id = Column(String, index=True, nullable=False)

    input_data = Column(JSON, nullable=False)
    prediction = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PipelineGraph(Base):
    __tablename__ = "pipeline_graphs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    nodes = Column(JSON, nullable=False)
    edges = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DriftReport(Base):
    __tablename__ = "drift_reports"

    report_id = Column(String, primary_key=True, index=True)
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
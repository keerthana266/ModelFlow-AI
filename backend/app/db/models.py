import uuid

from sqlalchemy import Column, String, Float, DateTime, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


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


class PipelineGraph(Base):
    __tablename__ = "pipeline_graphs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    nodes = Column(JSON, nullable=False)
    edges = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
# app/ml/trainer.py

from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from flaml import AutoML

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.db import models
from app.db.database import Base, SessionLocal, engine
from app.ml.detector import detect_problem_type
from app.ml.preprocessor import build_preprocessor, get_column_groups
from app.services.storage import MODEL_DIR, generate_id, save_experiment_result
from app.worker import celery_app


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1_score": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
    }


def calculate_regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2_score(y_true, y_pred)), 4),
    }


def safe_train_test_split(X, y, problem_type: str):
    stratify = None

    if problem_type == "classification":
        class_counts = y.value_counts()

        if len(class_counts) > 1 and class_counts.min() >= 2:
            stratify = y

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )


def validate_training_data(df: pd.DataFrame, target_column: str) -> None:
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the dataset."
        )

    if df.shape[0] < 10:
        raise ValueError("Dataset must contain at least 10 rows.")

    if df.shape[1] < 2:
        raise ValueError(
            "Dataset must contain at least one feature column and one target column."
        )


def calculate_shap_importance(
    preprocessor,
    best_model_instance,
    X_train_transformed,
    X_test_transformed,
) -> dict:
    try:
        feature_names = preprocessor.get_feature_names_out()

        background_size = min(100, X_train_transformed.shape[0])
        background_data = shap.sample(X_train_transformed, background_size)

        explainer = shap.Explainer(best_model_instance, background_data)
        shap_values = explainer(X_test_transformed)

        if len(shap_values.values.shape) == 3:
            mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

        feature_importance = {
            name.replace("num__", "").replace("cat__", ""): round(float(value), 4)
            for name, value in zip(feature_names, mean_abs_shap)
        }

        return dict(
            sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    except Exception as error:
        return {
            "error": f"SHAP calculation not supported for this model. {str(error)}"
        }


def log_training_run_to_mlflow(
    model_path,
    result: dict,
    metrics: dict,
    feature_importance: dict,
    best_estimator_name: str,
    time_budget: int,
) -> dict:
    """
    Logs training metadata, metrics, feature importance, and the saved .joblib
    model artifact to MLflow.

    This avoids MLflow sklearn serialization issues on Windows.
    """

    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        run_name = f"{result['experiment_id']}__{best_estimator_name}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("mlflow.source.name", "AutoML Forge Celery Worker")
            mlflow.set_tag("project", "AutoML Forge")
            mlflow.set_tag("run_type", "automl_training")
            mlflow.set_tag("training_status", "completed")

            mlflow.log_params(
                {
                    "experiment_id": result["experiment_id"],
                    "best_model_id": result["best_model_id"],
                    "best_estimator": best_estimator_name,
                    "problem_type": result["problem_type"],
                    "target_column": result["target_column"],
                    "rows_used": result["rows_used"],
                    "total_columns": result["total_columns"],
                    "main_metric": result["main_metric"],
                    "time_budget": time_budget,
                    "num_features": len(result["feature_columns"]),
                }
            )

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, float(metric_value))

            mlflow.log_dict(
                {
                    "feature_columns": result["feature_columns"],
                    "numerical_columns": result["numerical_columns"],
                    "categorical_columns": result["categorical_columns"],
                },
                "columns.json",
            )

            mlflow.log_dict(
                feature_importance,
                "feature_importance.json",
            )

            try:
                mlflow.log_artifact(
                    local_path=str(model_path),
                    artifact_path="model_package",
                )
                mlflow.set_tag("model_log_status", "success")
                model_logged = True

            except Exception as error:
                mlflow.set_tag("model_log_status", "failed")
                mlflow.set_tag("model_log_error", str(error)[:500])
                model_logged = False

            return {
                "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
                "mlflow_experiment_name": settings.MLFLOW_EXPERIMENT_NAME,
                "mlflow_run_id": run.info.run_id,
                "mlflow_artifact_uri": run.info.artifact_uri,
                "mlflow_model_logged": model_logged,
            }

    except Exception as error:
        return {
            "mlflow_error": str(error),
        }


def save_training_result_to_database(result: dict, model_path: str) -> None:
    db = SessionLocal()

    try:
        Base.metadata.create_all(bind=engine)

        best_model = result["leaderboard"][0]

        experiment_record = models.ExperimentResult(
            experiment_id=result["experiment_id"],
            problem_type=result["problem_type"],
            target_column=result["target_column"],
            rows_used=result["rows_used"],
            best_model_id=result["best_model_id"],
            main_metric=result["main_metric"],
            best_score=result["best_score"],
            feature_importance=result["feature_importance"],
            leaderboard=result["leaderboard"],
        )

        model_record = models.TrainedModel(
            model_id=result["best_model_id"],
            experiment_id=result["experiment_id"],
            model_name=best_model["model_name"],
            problem_type=result["problem_type"],
            target_column=result["target_column"],
            main_metric=result["main_metric"],
            score=result["best_score"],
            metrics=best_model["metrics"],
            feature_columns=result["feature_columns"],
            numerical_columns=result["numerical_columns"],
            categorical_columns=result["categorical_columns"],
            model_path=model_path,
        )

        db.merge(experiment_record)
        db.merge(model_record)
        db.commit()

    except Exception as error:
        db.rollback()
        print(f"DATABASE SAVE FAILED: {error}")

    finally:
        db.close()


@celery_app.task(bind=True, name="train_automl_task")
def train_automl(
    self,
    dataset_path_str: str,
    target_column: str,
    time_budget: int = 30,
) -> dict:
    time_budget = max(10, min(int(time_budget), 300))

    self.update_state(
        state="PROGRESS",
        meta={"message": "Loading and validating dataset..."},
    )

    dataset_path = Path(dataset_path_str)
    df = pd.read_csv(dataset_path)

    validate_training_data(df, target_column)

    df = df.dropna(subset=[target_column])

    if df.shape[0] < 10:
        raise ValueError(
            "Dataset must contain at least 10 rows after removing empty target values."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.empty:
        raise ValueError("There are no feature columns available for training.")

    problem_type = detect_problem_type(y)
    column_groups = get_column_groups(X)

    X_train, X_test, y_train, y_test = safe_train_test_split(
        X,
        y,
        problem_type,
    )

    self.update_state(
        state="PROGRESS",
        meta={"message": "Preprocessing features..."},
    )

    preprocessor = build_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    self.update_state(
        state="PROGRESS",
        meta={"message": "FLAML is actively optimizing models. Please wait..."},
    )

    automl = AutoML()

    automl_settings = {
        "time_budget": time_budget,
        "metric": "f1" if problem_type == "classification" else "r2",
        "task": problem_type,
        "verbose": 0,
    }

    automl.fit(
        X_train=X_train_transformed,
        y_train=y_train,
        **automl_settings,
    )

    best_estimator_name = automl.best_estimator
    best_model_instance = automl.model.estimator

    self.update_state(
        state="PROGRESS",
        meta={"message": "Packaging the optimized pipeline..."},
    )

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", best_model_instance),
        ]
    )

    self.update_state(
        state="PROGRESS",
        meta={"message": "Calculating SHAP feature importance..."},
    )

    feature_importance = calculate_shap_importance(
        preprocessor=preprocessor,
        best_model_instance=best_model_instance,
        X_train_transformed=X_train_transformed,
        X_test_transformed=X_test_transformed,
    )

    predictions = final_pipeline.predict(X_test)

    if problem_type == "classification":
        metrics = calculate_classification_metrics(y_test, predictions)
        main_metric = "f1_score"

    else:
        metrics = calculate_regression_metrics(y_test, predictions)
        main_metric = "r2_score"

    score = metrics[main_metric]

    model_id = generate_id("model")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{model_id}.joblib"

    model_package = {
        "model_id": model_id,
        "model_name": f"Optimized {best_estimator_name.upper()}",
        "problem_type": problem_type,
        "target_column": target_column,
        "feature_columns": X.columns.tolist(),
        "numerical_columns": column_groups["numerical_columns"],
        "categorical_columns": column_groups["categorical_columns"],
        "pipeline": final_pipeline,
        "metrics": metrics,
        "time_budget": time_budget,
        "created_at": utc_now_iso(),
    }

    joblib.dump(model_package, model_path)

    leaderboard = [
        {
            "model_id": model_id,
            "model_name": f"FLAML Tuned: {best_estimator_name.upper()}",
            "status": "success",
            "metrics": metrics,
            "main_metric": main_metric,
            "score": score,
            "is_best_model": True,
        }
    ]

    result = {
        "experiment_id": generate_id("experiment"),
        "problem_type": problem_type,
        "target_column": target_column,
        "rows_used": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "feature_columns": X.columns.tolist(),
        "numerical_columns": column_groups["numerical_columns"],
        "categorical_columns": column_groups["categorical_columns"],
        "main_metric": main_metric,
        "best_score": score,
        "best_model_id": model_id,
        "feature_importance": feature_importance,
        "leaderboard": leaderboard,
        "time_budget": time_budget,
        "created_at": utc_now_iso(),
    }

    self.update_state(
        state="PROGRESS",
        meta={"message": "Logging experiment to MLflow..."},
    )

    mlflow_info = log_training_run_to_mlflow(
         model_path=model_path,
         result=result,
         metrics=metrics,
         feature_importance=feature_importance,
         best_estimator_name=best_estimator_name,
         time_budget=time_budget,
    )

    result.update(mlflow_info)

    if "mlflow_run_id" in mlflow_info:
        result["leaderboard"][0]["mlflow_run_id"] = mlflow_info["mlflow_run_id"]
        result["leaderboard"][0]["mlflow_artifact_uri"] = mlflow_info.get(
            "mlflow_artifact_uri"
        )

    save_experiment_result(result["experiment_id"], result)
    save_training_result_to_database(result, str(model_path))

    return result
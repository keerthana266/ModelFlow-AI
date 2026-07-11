# app/ml/trainer.py

from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import shap
from flaml import AutoML

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from app.services.artifact_store import (
    upload_experiment_artifact,
    upload_model_artifact,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from app.core.config import settings
from app.db import models
from app.db.database import Base, SessionLocal, engine
from app.db.migrations import ensure_project_scope_columns
from app.ml.detector import detect_problem_type
from app.ml.drift import save_reference_data
from app.ml.preprocessor import build_preprocessor, get_column_groups
from app.services.storage import MODEL_DIR, generate_id, save_experiment_result
from app.worker import celery_app


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_dense_matrix(X):
    if sp.issparse(X):
        return X.toarray()

    return X


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


def build_optuna_cv(y, problem_type: str):
    if problem_type == "classification":
        class_counts = y.value_counts()
        min_class_count = int(class_counts.min())

        if min_class_count >= 2:
            splits = min(3, min_class_count)

            return StratifiedKFold(
                n_splits=splits,
                shuffle=True,
                random_state=42,
            )

        return KFold(
            n_splits=2,
            shuffle=True,
            random_state=42,
        )

    splits = min(3, max(2, len(y) // 3))

    return KFold(
        n_splits=splits,
        shuffle=True,
        random_state=42,
    )


def build_optuna_estimator(trial, problem_type: str):
    if problem_type == "classification":
        model_type = trial.suggest_categorical(
            "model_type",
            [
                "random_forest",
                "extra_trees",
                "gradient_boosting",
                "logistic_regression",
            ],
        )

        if model_type == "random_forest":
            return (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
                    max_depth=trial.suggest_int("rf_max_depth", 2, 20),
                    min_samples_split=trial.suggest_int(
                        "rf_min_samples_split", 2, 10
                    ),
                    min_samples_leaf=trial.suggest_int(
                        "rf_min_samples_leaf", 1, 5
                    ),
                    max_features=trial.suggest_categorical(
                        "rf_max_features", ["sqrt", "log2", None]
                    ),
                    random_state=42,
                    n_jobs=-1,
                ),
            )

        if model_type == "extra_trees":
            return (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=trial.suggest_int("et_n_estimators", 50, 300),
                    max_depth=trial.suggest_int("et_max_depth", 2, 20),
                    min_samples_split=trial.suggest_int(
                        "et_min_samples_split", 2, 10
                    ),
                    min_samples_leaf=trial.suggest_int(
                        "et_min_samples_leaf", 1, 5
                    ),
                    max_features=trial.suggest_categorical(
                        "et_max_features", ["sqrt", "log2", None]
                    ),
                    random_state=42,
                    n_jobs=-1,
                ),
            )

        if model_type == "gradient_boosting":
            return (
                "gradient_boosting",
                GradientBoostingClassifier(
                    n_estimators=trial.suggest_int("gb_n_estimators", 50, 200),
                    learning_rate=trial.suggest_float(
                        "gb_learning_rate", 0.01, 0.3, log=True
                    ),
                    max_depth=trial.suggest_int("gb_max_depth", 2, 8),
                    subsample=trial.suggest_float("gb_subsample", 0.6, 1.0),
                    random_state=42,
                ),
            )

        return (
            "logistic_regression",
            LogisticRegression(
                C=trial.suggest_float("lr_C", 0.01, 100.0, log=True),
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            ),
        )

    model_type = trial.suggest_categorical(
        "model_type",
        [
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "ridge",
            "elasticnet",
        ],
    )

    if model_type == "random_forest":
        return (
            "random_forest",
            RandomForestRegressor(
                n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
                max_depth=trial.suggest_int("rf_max_depth", 2, 20),
                min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical(
                    "rf_max_features", ["sqrt", "log2", None]
                ),
                random_state=42,
                n_jobs=-1,
            ),
        )

    if model_type == "extra_trees":
        return (
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=trial.suggest_int("et_n_estimators", 50, 300),
                max_depth=trial.suggest_int("et_max_depth", 2, 20),
                min_samples_split=trial.suggest_int("et_min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical(
                    "et_max_features", ["sqrt", "log2", None]
                ),
                random_state=42,
                n_jobs=-1,
            ),
        )

    if model_type == "gradient_boosting":
        return (
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=trial.suggest_int("gb_n_estimators", 50, 200),
                learning_rate=trial.suggest_float(
                    "gb_learning_rate", 0.01, 0.3, log=True
                ),
                max_depth=trial.suggest_int("gb_max_depth", 2, 8),
                subsample=trial.suggest_float("gb_subsample", 0.6, 1.0),
                random_state=42,
            ),
        )

    if model_type == "ridge":
        return (
            "ridge",
            Ridge(
                alpha=trial.suggest_float("ridge_alpha", 0.01, 100.0, log=True),
                random_state=42,
            ),
        )

    return (
        "elasticnet",
        ElasticNet(
            alpha=trial.suggest_float("en_alpha", 0.01, 100.0, log=True),
            l1_ratio=trial.suggest_float("en_l1_ratio", 0.05, 0.95),
            random_state=42,
            max_iter=2000,
        ),
    )


def run_optuna_tuning(
    X_train_transformed,
    y_train,
    problem_type: str,
    time_budget: int,
):
    X_train_dense = to_dense_matrix(X_train_transformed)

    scoring = "f1_weighted" if problem_type == "classification" else "r2"
    cv = build_optuna_cv(y_train, problem_type)

    study = optuna.create_study(direction="maximize")

    def objective(trial):
        _, estimator = build_optuna_estimator(trial, problem_type)

        try:
            scores = cross_val_score(
                estimator,
                X_train_dense,
                y_train,
                scoring=scoring,
                cv=cv,
                n_jobs=1,
                error_score="raise",
            )

            return float(scores.mean())

        except Exception:
            return -1e9

    study.optimize(
        objective,
        timeout=time_budget,
        n_trials=50,
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    fixed_trial = optuna.trial.FixedTrial(best_trial.params)

    best_estimator_name, best_estimator = build_optuna_estimator(
        fixed_trial,
        problem_type,
    )

    best_estimator.fit(X_train_dense, y_train)

    return {
        "best_estimator_name": best_estimator_name,
        "best_model_instance": best_estimator,
        "optuna_best_value": float(best_trial.value),
        "optuna_best_params": best_trial.params,
        "optuna_trials": len(study.trials),
    }


def log_training_run_to_mlflow(
    model_path,
    result: dict,
    metrics: dict,
    feature_importance: dict,
    best_estimator_name: str,
    time_budget: int,
) -> dict:
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        run_name = f"{result['experiment_id']}__{best_estimator_name}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("mlflow.source.name", "AutoML Forge Celery Worker")
            mlflow.set_tag("project", "AutoML Forge")
            mlflow.set_tag("run_type", "automl_training")
            mlflow.set_tag("training_status", "completed")
            mlflow.set_tag("tuning_mode", result.get("tuning_mode", "flaml"))
            mlflow.set_tag("automl_owner_id", result.get("owner_id") or "legacy")
            mlflow.set_tag("automl_project_id", result.get("project_id") or "legacy")

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
                    "tuning_mode": result.get("tuning_mode", "flaml"),
                    "num_features": len(result["feature_columns"]),
                }
            )

            optuna_info = result.get("optuna_info") or {}

            if optuna_info:
                mlflow.log_params(
                    {
                        "optuna_trials": optuna_info.get("optuna_trials"),
                        "optuna_best_value": optuna_info.get("optuna_best_value"),
                    }
                )

                mlflow.log_dict(
                    optuna_info,
                    "optuna_info.json",
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
    model_id = result["best_model_id"]
    experiment_id = result["experiment_id"]

    artifact_store_info = {
        "enabled": False,
        "backend": "local",
    }

    try:
        model_artifact = upload_model_artifact(
            local_model_path=model_path,
            experiment_id=experiment_id,
            model_id=model_id,
        )

        artifact_store_info = {
            "enabled": model_artifact.get("enabled", False),
            "backend": model_artifact.get("backend", "minio"),
            "model_artifact": model_artifact,
        }

        result["artifact_store"] = artifact_store_info

        try:
            model_package = joblib.load(model_path)
            model_package["artifact_store"] = artifact_store_info
            joblib.dump(model_package, model_path)

            refreshed_model_artifact = upload_model_artifact(
                local_model_path=model_path,
                experiment_id=experiment_id,
                model_id=model_id,
            )

            artifact_store_info["model_artifact"] = refreshed_model_artifact
            result["artifact_store"] = artifact_store_info

        except Exception as package_error:
            artifact_store_info["model_package_metadata_error"] = str(package_error)
            result["artifact_store"] = artifact_store_info

        experiment_artifact = upload_experiment_artifact(
            result=result,
            experiment_id=experiment_id,
        )

        artifact_store_info["experiment_artifact"] = experiment_artifact
        result["artifact_store"] = artifact_store_info

        save_experiment_result(experiment_id, result)

    except Exception as artifact_error:
        artifact_store_info = {
            "enabled": True,
            "backend": "minio",
            "status": "failed",
            "error": str(artifact_error),
        }

        result["artifact_store"] = artifact_store_info
        save_experiment_result(experiment_id, result)

    db = SessionLocal()

    try:
        Base.metadata.create_all(bind=engine)
        ensure_project_scope_columns()

        best_model = result["leaderboard"][0]

        experiment_record = models.ExperimentResult(
            experiment_id=result["experiment_id"],
            owner_id=result.get("owner_id"),
            project_id=result.get("project_id"),
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
            owner_id=result.get("owner_id"),
            project_id=result.get("project_id"),
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
    tuning_mode: str = "flaml",
    owner_id: str | None = None,
    project_id: str | None = None,
) -> dict:
    time_budget = max(10, min(int(time_budget), 300))

    if tuning_mode not in ["flaml", "optuna"]:
        tuning_mode = "flaml"

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

    optuna_info = {}

    if tuning_mode == "optuna":
        self.update_state(
            state="PROGRESS",
            meta={
                "message": "Optuna is tuning hyperparameters across candidate models..."
            },
        )

        optuna_result = run_optuna_tuning(
            X_train_transformed=X_train_transformed,
            y_train=y_train,
            problem_type=problem_type,
            time_budget=time_budget,
        )

        best_estimator_name = optuna_result["best_estimator_name"]
        best_model_instance = optuna_result["best_model_instance"]

        optuna_info = {
            "optuna_best_value": optuna_result["optuna_best_value"],
            "optuna_best_params": optuna_result["optuna_best_params"],
            "optuna_trials": optuna_result["optuna_trials"],
        }

    else:
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

    if tuning_mode == "optuna":
        final_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "to_dense",
                    FunctionTransformer(
                        to_dense_matrix,
                        accept_sparse=True,
                    ),
                ),
                ("model", best_model_instance),
            ]
        )

    else:
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

    shap_train_data = (
        to_dense_matrix(X_train_transformed)
        if tuning_mode == "optuna"
        else X_train_transformed
    )

    shap_test_data = (
        to_dense_matrix(X_test_transformed)
        if tuning_mode == "optuna"
        else X_test_transformed
    )

    feature_importance = calculate_shap_importance(
        preprocessor=preprocessor,
        best_model_instance=best_model_instance,
        X_train_transformed=shap_train_data,
        X_test_transformed=shap_test_data,
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
    reference_data_path = save_reference_data(model_id, X)

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
        "tuning_mode": tuning_mode,
        "optuna_info": optuna_info,
        "reference_data_path": str(reference_data_path),
        "owner_id": owner_id,
        "project_id": project_id,
        "created_at": utc_now_iso(),
    }

    joblib.dump(model_package, model_path)

    leaderboard_model_name = (
        f"Optuna Tuned: {best_estimator_name.upper()}"
        if tuning_mode == "optuna"
        else f"FLAML Tuned: {best_estimator_name.upper()}"
    )

    leaderboard = [
        {
            "model_id": model_id,
            "model_name": leaderboard_model_name,
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
        "tuning_mode": tuning_mode,
        "optuna_info": optuna_info,
        "owner_id": owner_id,
        "project_id": project_id,
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
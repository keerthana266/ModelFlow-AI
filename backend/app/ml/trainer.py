from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.ml.detector import detect_problem_type
from app.ml.preprocessor import build_preprocessor, get_column_groups
from app.services.storage import MODEL_DIR, generate_id, save_experiment_result
from app.worker import celery_app


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_classification_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            random_state=42,
        ),
    }


def get_regression_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=42,
        ),
    }


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

@celery_app.task(bind=True, name="train_automl_task")
def train_automl(self, dataset_path_str: str, target_column: str) -> dict:
    
    self.update_state(state='PROGRESS', meta={'message': 'Loading dataset...'})
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

    if problem_type == "classification":
        models = get_classification_models()
        main_metric = "f1_score"
    else:
        models = get_regression_models()
        main_metric = "r2_score"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    leaderboard = []
    best_model_id = None
    best_score = -999999.0

    total_models = len(models)
    current_model_idx = 0

    for model_name, model in models.items():
        current_model_idx += 1
        self.update_state(
            state='PROGRESS', 
            meta={
                'message': f'Training {model_name} ({current_model_idx}/{total_models})...',
                'current': current_model_idx,
                'total': total_models
            }
        )
        
        try:
            preprocessor = build_preprocessor(X_train)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            if problem_type == "classification":
                metrics = calculate_classification_metrics(y_test, predictions)
            else:
                metrics = calculate_regression_metrics(y_test, predictions)

            score = metrics[main_metric]

            model_id = generate_id("model")
            model_path = MODEL_DIR / f"{model_id}.joblib"

            model_package = {
                "model_id": model_id,
                "model_name": model_name,
                "problem_type": problem_type,
                "target_column": target_column,
                "feature_columns": X.columns.tolist(),
                "numerical_columns": column_groups["numerical_columns"],
                "categorical_columns": column_groups["categorical_columns"],
                "pipeline": pipeline,
                "metrics": metrics,
                "created_at": utc_now_iso(),
            }

            joblib.dump(model_package, model_path)

            if score > best_score:
                best_score = score
                best_model_id = model_id

            leaderboard.append(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "status": "success",
                    "metrics": metrics,
                    "main_metric": main_metric,
                    "score": score,
                    "model_path": str(model_path),
                }
            )

        except Exception as error:
            leaderboard.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "error": str(error),
                }
            )

    leaderboard = sorted(
        leaderboard,
        key=lambda item: item.get("score", -999999.0),
        reverse=True,
    )

    for item in leaderboard:
        item["is_best_model"] = item.get("model_id") == best_model_id

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
        "best_model_id": best_model_id,
        "leaderboard": leaderboard,
        "created_at": utc_now_iso(),
    }
    
    # Save the result to disk from the worker before returning
    save_experiment_result(result["experiment_id"], result)
    
    return result
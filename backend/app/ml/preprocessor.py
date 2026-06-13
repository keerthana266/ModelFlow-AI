import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_DTYPES = ["int64", "float64", "int32", "float32"]


def create_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_column_groups(X: pd.DataFrame) -> dict:
    numerical_columns = X.select_dtypes(include=NUMERIC_DTYPES).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=NUMERIC_DTYPES).columns.tolist()

    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
    }


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    column_groups = get_column_groups(X)

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", create_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, column_groups["numerical_columns"]),
            ("cat", categorical_pipeline, column_groups["categorical_columns"]),
        ],
        remainder="drop",
    )

    return preprocessor
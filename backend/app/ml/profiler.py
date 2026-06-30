# app/ml/profiler.py

from pathlib import Path

import pandas as pd


def profile_csv_dataset(dataset_path: Path, sample_size: int = 5) -> dict:
    df = pd.read_csv(dataset_path)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {
            column: int(value)
            for column, value in df.isna().sum().items()
        },
        "unique_values": {
            column: int(value)
            for column, value in df.nunique(dropna=True).items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "sample_rows": df.head(sample_size).to_dict(orient="records"),
    }
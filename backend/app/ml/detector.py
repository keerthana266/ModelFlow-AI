import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def detect_problem_type(y: pd.Series) -> str:
    """
    Detect whether the target column is classification or regression.

    Sprint 1 logic:
    - Text/category/bool target = classification
    - Numeric target with 2 unique values = classification
    - Numeric target with few repeated integer labels = classification
    - Continuous numeric target = regression
    """

    y = y.dropna()

    if y.empty:
        raise ValueError("Target column is empty after removing missing values.")

    unique_count = y.nunique()
    total_count = len(y)
    unique_ratio = unique_count / total_count

    if is_bool_dtype(y):
        return "classification"

    if not is_numeric_dtype(y):
        return "classification"

    if unique_count <= 2:
        return "classification"

    if unique_count <= 20 and unique_ratio <= 0.2:
        return "classification"

    return "regression"
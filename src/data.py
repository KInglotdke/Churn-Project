import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    DROP_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def prepare_features_and_target(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    columns_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    X = df.drop(columns=columns_to_drop + [TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_column_types(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numerical_cols
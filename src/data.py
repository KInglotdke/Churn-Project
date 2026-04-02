import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TARGET_COLUMN, DROP_COLUMNS, RANDOM_STATE, TEST_SIZE


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def prepare_features_and_target(df):
    X = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS, errors="ignore")
    y = df[TARGET_COLUMN].copy()
    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_feature_types(X):
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    return categorical_features, numerical_features
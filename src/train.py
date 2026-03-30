import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from src.config import MODELS_DIR, REPORTS_DIR
from src.data import load_data, prepare_features_and_target, split_data, get_column_types
from src.utils import ensure_dir, save_json


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols, numerical_cols = get_column_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    report = classification_report(y_test, y_pred, output_dict=True)
    return metrics, report


def train_and_save_model() -> None:
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)

    df = load_data()
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_model_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    metrics, report = evaluate_model(pipeline, X_test, y_test)

    model_path = MODELS_DIR / "churn_model.joblib"
    metrics_path = REPORTS_DIR / "metrics.json"
    report_path = REPORTS_DIR / "classification_report.json"

    joblib.dump(pipeline, model_path)
    save_json(metrics, metrics_path)
    save_json(report, report_path)

    print("Training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
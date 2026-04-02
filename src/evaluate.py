import matplotlib

from src.config import BEST_MODEL_ASCENDING, BEST_MODEL_METRIC
matplotlib.use("Agg")

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
    balanced_accuracy_score,
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        results["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        results["roc_auc"] = None

    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    return results, y_pred, y_proba, report_dict


def save_metrics_table(results_list, output_csv, output_json):
    df = pd.DataFrame(results_list).sort_values(by=BEST_MODEL_METRIC, ascending=BEST_MODEL_ASCENDING)
    df.to_csv(output_csv, index=False)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=4)

    return df


def save_classification_report(report_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)


def save_confusion_matrix(y_test, y_pred, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_roc_curve(model, X_test, y_test, output_path):
    if not hasattr(model, "predict_proba"):
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_precision_recall_curve(model, X_test, y_test, output_path):
    if not hasattr(model, "predict_proba"):
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_model(model, output_path):
    joblib.dump(model, output_path)
from pathlib import Path

from src.config import (
    DATA_PATH,
    REPORTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    BENCHMARK_CSV,
    BENCHMARK_JSON,
    BEST_MODEL_FILENAME,
)
from src.data import load_data, prepare_features_and_target, split_data, get_feature_types
from src.models import build_model_pipelines
from src.evaluate import (
    evaluate_model,
    save_metrics_table,
    save_classification_report,
    save_confusion_matrix,
    save_roc_curve,
    save_precision_recall_curve,
    save_model,
)
from src.utils import ensure_directories, setup_logging


def run_training_pipeline():
    ensure_directories([REPORTS_DIR, METRICS_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR])

    logger = setup_logging(LOGS_DIR / "training.log")
    logger.info("Starting bank churn training pipeline")

    logger.info("Loading dataset from %s", DATA_PATH)
    df = load_data(DATA_PATH)
    logger.info("Dataset loaded with shape: %s", df.shape)

    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    logger.info("Total missing values: %d", missing_values)
    logger.info("Duplicate rows: %d", duplicate_rows)

    X, y = prepare_features_and_target(df)
    logger.info("Prepared features and target")
    logger.info("Feature matrix shape: %s", X.shape)
    logger.info("Target distribution: %s", y.value_counts(normalize=True).to_dict())

    categorical_features, numerical_features = get_feature_types(X)
    logger.info("Numerical features (%d): %s", len(numerical_features), numerical_features)
    logger.info("Categorical features (%d): %s", len(categorical_features), categorical_features)

    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)

    model_pipelines = build_model_pipelines(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        random_state=RANDOM_STATE,
    )

    benchmark_results = []
    fitted_models = {}

    for model_name, pipeline in model_pipelines.items():
        logger.info("Training model: %s", model_name)

        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        metrics, y_pred, y_proba, report_dict = evaluate_model(pipeline, X_test, y_test)
        metrics["model"] = model_name
        benchmark_results.append(metrics)

        logger.info("Metrics for %s: %s", model_name, metrics)

        save_classification_report(
            report_dict,
            REPORTS_DIR / f"{model_name}_classification_report.json",
        )
        save_confusion_matrix(
            y_test,
            y_pred,
            FIGURES_DIR / f"{model_name}_confusion_matrix.png",
        )
        save_roc_curve(
            pipeline,
            X_test,
            y_test,
            FIGURES_DIR / f"{model_name}_roc_curve.png",
        )
        save_precision_recall_curve(
            pipeline,
            X_test,
            y_test,
            FIGURES_DIR / f"{model_name}_pr_curve.png",
        )

    benchmark_df = save_metrics_table(
        benchmark_results,
        BENCHMARK_CSV,
        BENCHMARK_JSON,
    )

    best_model_name = benchmark_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    save_model(best_model, MODELS_DIR / BEST_MODEL_FILENAME)

    logger.info("Best model selected: %s", best_model_name)
    logger.info("Benchmark saved to: %s and %s", BENCHMARK_CSV, BENCHMARK_JSON)
    logger.info("Best model saved to: %s", MODELS_DIR / BEST_MODEL_FILENAME)
    logger.info("Training pipeline completed successfully")

    return benchmark_df, best_model_name
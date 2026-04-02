from pathlib import Path

# Project root = parent of /src
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "bank_churn.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

TARGET_COLUMN = "Exited"

DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

RANDOM_STATE = 80
TEST_SIZE = 0.2

BEST_MODEL_METRIC = "roc_auc"
BEST_MODEL_ASCENDING = False
BEST_MODEL_FILENAME = "best_model.pkl"
BENCHMARK_CSV = METRICS_DIR / "model_benchmark.csv"
BENCHMARK_JSON = METRICS_DIR / "model_benchmark.json"
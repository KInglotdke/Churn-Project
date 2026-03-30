from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "bank_churn.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

TARGET_COLUMN = "Exited"

# Common columns to drop if present
DROP_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname"
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
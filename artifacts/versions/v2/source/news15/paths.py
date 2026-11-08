from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw" / "isot"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"


def ensure_directories():
    for directory in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

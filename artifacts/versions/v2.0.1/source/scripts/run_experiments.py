"""Run the bounded reproducible experiments after data preparation/training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Supports both `python scripts/run_experiments.py` and module execution.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from news15.experiments import run_experiments
from news15.paths import REPORTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run topic 15 bias, noise, source ablation, SMOTE and 2D boundary experiments.")
    parser.add_argument("--sample-size", type=int, default=1200, help="Maximum held-out sample budget (internally capped at 1000).")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR, help="CSV and JSON report directory.")
    arguments = parser.parse_args()
    results = run_experiments(output_dir=arguments.output_dir, sample_size=arguments.sample_size)
    print(json.dumps({
        "report_file": results["report_file"],
        "metadata": results["metadata"],
        "baseline_metrics": results["baseline_metrics"],
        "smote_results": results["smote"]["results"],
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

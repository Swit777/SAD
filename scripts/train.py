import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.training import train_project, refine_project


def main():
    parser = argparse.ArgumentParser(description="Train the current English news model with grouped data and independent evaluation.")
    parser.add_argument('--refine',action='store_true',help='Refine format-specific calibration and stacking with fixed base estimators; old external data are reused diagnostics.')
    args=parser.parse_args()
    bundle = refine_project() if args.refine else train_project()
    print(json.dumps({
        "selected_model": bundle["metadata"]["selected_model"],
        "threshold": bundle["threshold"],
        "test_metrics": bundle["metadata"]["test_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()

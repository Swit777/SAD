import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.model import train_models


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ISOT classifiers.")
    parser.add_argument("--max-features", type=int, default=40000)
    args = parser.parse_args()
    bundle = train_models(max_features=args.max_features)
    print(json.dumps({
        "selected_model": bundle["metadata"]["selected_model"],
        "threshold": bundle["threshold"],
        "test_metrics": bundle["metadata"]["test_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()

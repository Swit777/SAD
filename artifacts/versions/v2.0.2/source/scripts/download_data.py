"""Download original ISOT data and prepare the chronological research corpus."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news15.data import download_isot, prepare_isot, dataset_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download the original archive again.")
    args = parser.parse_args()
    print("Downloading/verifying original ISOT dataset...", flush=True)
    print(download_isot(force=args.force), flush=True)
    prepare_isot()
    print(json.dumps(dataset_summary(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

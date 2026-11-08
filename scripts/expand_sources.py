"""Download and import additional public datasets into corpus_v4."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.multisource import import_sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-articles", type=int, default=250000)
    parser.add_argument("--domain-cap", type=int, default=10000)
    parser.add_argument("--resume", action="store_true", help="Verify and resume the last completed partial output.")
    args = parser.parse_args()
    import_sources(args.target_articles, args.domain_cap, resume=args.resume)

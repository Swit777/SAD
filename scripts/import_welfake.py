"""Download, normalize and deduplicate WELFake without changing frozen models."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.corpus_expansion import download_welfake

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    source = download_welfake()
    print(source, flush=True)
    if not args.download_only:
        from news15.corpus_expansion import import_welfake
        import_welfake(source)

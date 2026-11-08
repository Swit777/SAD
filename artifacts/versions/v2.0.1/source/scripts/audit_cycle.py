"""Run the preregistered methodological cycle; external data never tune models."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.audit import run_audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=19)
    parser.add_argument("--policy-bootstrap", type=int, default=60)
    parser.add_argument("--no-activate", action="store_true")
    parser.add_argument("--reproduce", action="store_true", help="Write a separate non-activated reproduction; external set is already seen.")
    args = parser.parse_args()
    if args.permutations < 1 or args.policy_bootstrap < 1:
        parser.error("Controls and bootstrap repetition counts must be positive.")
    if args.reproduce:
        from datetime import datetime, timezone
        import news15.audit as audit
        from news15.paths import REPORTS_DIR
        tag = "v2-reproduction-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        audit.AUDIT_DIR = REPORTS_DIR / "reproductions" / tag
        run_audit(args.permutations, args.policy_bootstrap, False, tag)
    else:
        run_audit(args.permutations, args.policy_bootstrap, not args.no_activate)

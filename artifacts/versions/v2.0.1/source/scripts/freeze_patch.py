"""Freeze a tested source-only patch without changing the scientific model."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.paths import PROJECT_DIR, ARTIFACTS_DIR
from news15.data import file_sha256
from news15.audit import write_json

parent = ARTIFACTS_DIR / "versions" / "v2"
target = ARTIFACTS_DIR / "versions" / "v2.0.1"
if target.exists():
    raise SystemExit("Patch version exists and will not be overwritten.")
manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
assert file_sha256(ARTIFACTS_DIR / "model.joblib") == manifest["model_sha256"], "Model changed; not a source-only patch."
target.mkdir(parents=True)
shutil.copy2(parent / "model.joblib", target / "model.joblib")
shutil.copytree(parent / "reports", target / "reports")
files = sorted((PROJECT_DIR / "news15").glob("*.py")) + sorted((PROJECT_DIR / "scripts").glob("*.py")) + sorted((PROJECT_DIR / "tests").glob("*.py"))
files += [PROJECT_DIR / "app.py", PROJECT_DIR / "README.md", PROJECT_DIR / "requirements.txt", PROJECT_DIR / "protocols" / "v2.json"]
source_hashes = {}
for path in files:
    relative = path.relative_to(PROJECT_DIR)
    dest = target / "source" / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    source_hashes[str(relative)] = file_sha256(path)
manifest.update({
    "version": "2.0.1", "model_version": "2.0.0", "parent": "v2",
    "amendment": "UI numpy import correction and active-version verifier. Classifier, calibration, thresholds, predictions and scientific reports unchanged.",
    "source_files": source_hashes, "finalized": True,
})
write_json(target / "manifest.json", manifest)
write_json(ARTIFACTS_DIR / "active_version.json", {"version": "2.0.1", "directory": "v2.0.1", "model_version": "2.0.0"})
print("Frozen source patch 2.0.1; model and scientific report hashes unchanged.")

"""Verify a frozen version; edits are detected rather than silently re-hashed."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.paths import PROJECT_DIR, ARTIFACTS_DIR, REPORTS_DIR
from news15.data import file_sha256
from news15.audit import committee_fingerprint
from news15.model import load_bundle

directory = ARTIFACTS_DIR / "versions" / "v2"
manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
checks = {
    "version_model": file_sha256(directory / "model.joblib") == manifest["model_sha256"],
    "active_model": file_sha256(ARTIFACTS_DIR / "model.joblib") == manifest["model_sha256"],
    "protocol": file_sha256(PROJECT_DIR / "protocols" / "v2.json") == manifest["protocol_sha256"],
    "report": file_sha256(REPORTS_DIR / "audit_v2" / "audit_report.json") == manifest["report_sha256"],
    "external_manifest": file_sha256(REPORTS_DIR / "audit_v2" / "external_manifest.json") == manifest["external_manifest_sha256"],
}
bundle = load_bundle(directory / "model.joblib")
checks["model_parameters"] = committee_fingerprint(bundle["consensus"]) == manifest["parameter_fingerprint"]
for relative, digest in manifest["source_files"].items():
    checks[f"source:{relative}"] = file_sha256(PROJECT_DIR / relative) == digest
    checks[f"snapshot:{relative}"] = file_sha256(directory / "source" / relative) == digest
for relative, digest in manifest.get("report_files", {}).items():
    checks[f"report_snapshot:{relative}"] = file_sha256(directory / "reports" / relative) == digest
external = json.loads((REPORTS_DIR / "audit_v2" / "external_manifest.json").read_text(encoding="utf-8"))
for entry in external["files"]:
    checks[f"raw:{entry['file']}"] = file_sha256(PROJECT_DIR / "data" / "raw" / "fakenewsnet" / entry["file"]) == entry["sha256"]
print(json.dumps(checks, ensure_ascii=False, indent=2))
if not all(checks.values()):
    raise SystemExit("Version verification failed. Create a new version instead of overwriting the manifest.")

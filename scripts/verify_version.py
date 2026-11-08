"""Verify a frozen version; edits are detected rather than silently re-hashed."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.paths import PROJECT_DIR, ARTIFACTS_DIR, REPORTS_DIR
from news15.data import file_sha256
from news15.audit import committee_fingerprint
from news15.model import load_bundle

pointer = ARTIFACTS_DIR / "active_version.json"
active = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else {}
if active.get("model_file"):
    bundle = load_bundle(PROJECT_DIR / active["model_file"])
    checks = {
        "current_model": file_sha256(PROJECT_DIR / active["model_file"]) == active["model_sha256"],
        "protocol": file_sha256(PROJECT_DIR / active["protocol_file"]) == active["protocol_sha256"],
        "training_report": file_sha256(PROJECT_DIR / active["report_file"]) == active["report_sha256"],
        "full_model_parameters": committee_fingerprint(bundle["consensus"]) == active["fingerprints"]["full"],
    }
    if bundle.get('short_consensus') is not None:
        checks['short_prose_parameters']=committee_fingerprint(bundle['short_consensus'])==active['fingerprints']['short_prose']
        checks['headline_parameters']=committee_fingerprint(bundle['headline_consensus'])==active['fingerprints']['headline']
    else:
        checks['short_model_parameters']=committee_fingerprint(bundle['headline_consensus'])==active['fingerprints']['short']
    report=json.loads((PROJECT_DIR/active["report_file"]).read_text(encoding="utf-8"))
    checks["parameters_frozen_before_external"] = report["freeze"]["fingerprints"] == active["fingerprints"] and not report["freeze"]["external_labels_opened"]
    for section in ("data_files", "source_files"):
        for relative,digest in active.get(section,{}).items():
            checks[f"{section}:{relative}"] = file_sha256(PROJECT_DIR/relative) == digest
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    if not all(checks.values()):
        raise SystemExit("Current project verification failed; inspect the changed file or model parameters.")
    raise SystemExit(0)
version_directory = json.loads(pointer.read_text(encoding="utf-8"))["directory"] if pointer.exists() else "v2"
directory = ARTIFACTS_DIR / "versions" / version_directory
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
for relative,digest in manifest.get('data_files',{}).items():
    checks[f'data:{relative}']=file_sha256(PROJECT_DIR/relative)==digest
for relative,digest in manifest.get('deliverables',{}).items():
    checks[f'deliverable:{relative}']=file_sha256(PROJECT_DIR/relative)==digest
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

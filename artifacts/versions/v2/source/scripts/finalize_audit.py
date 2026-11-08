"""Finalize immutable v2 after diagnostics; never refit or retune on external data."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import sys
from urllib.parse import urlsplit
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.audit import AUDIT_DIR, PROTOCOL_PATH, agreement_report, calibration_table, classification_with_interval, committee_fingerprint, create_audit_markdown, write_json
from news15.consensus import policy_statistics
from news15.data import file_sha256
from news15.model import load_bundle, save_bundle
from news15.paths import PROJECT_DIR, ARTIFACTS_DIR, MODEL_PATH


def macro_from_counts(counts):
    tn, fp, fn, tp = counts
    a = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    b = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0
    return float((a+b)/2)


def finalize():
    report_path = AUDIT_DIR / "audit_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    version_dir = ARTIFACTS_DIR / "versions" / "v2"
    old_manifest = version_dir / "manifest.json"
    if old_manifest.exists() and json.loads(old_manifest.read_text(encoding="utf-8")).get("finalized"):
        raise SystemExit("Version 2 is already frozen; use verify_version.py or create a separate reproduction.")
    bundle = load_bundle(version_dir / "model.joblib")
    frozen = json.loads((AUDIT_DIR / "frozen_before_external.json").read_text(encoding="utf-8"))
    assert committee_fingerprint(bundle["consensus"]) == frozen["after_fingerprint"]
    # Correct multiple-comparison interpretation of existing training-only ANOVA.
    features_path = AUDIT_DIR / "global_xai_anova.csv"
    features = pd.read_csv(features_path)
    p = features.anova_p_label.to_numpy()
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1].clip(0, 1)
    original_order = np.empty_like(q)
    original_order[order] = q
    features["anova_q_BH"] = original_order
    features.to_csv(features_path, index=False)
    report["circularity"]["anova_caution"] = "BH-adjusted q-values are training-only diagnostics. Correlated words/articles and source-derived targets prevent a causal or independent-truth interpretation."
    # Publisher-cluster bootstrap uses existing frozen external predictions.
    # It is a post-freeze uncertainty diagnostic, not another tuning opportunity.
    raws = []
    raw_labels = []
    for entry in report["external_manifest"]["files"]:
        raw = pd.read_csv(PROJECT_DIR / "data" / "raw" / "fakenewsnet" / entry["file"], keep_default_na=False)
        raw_labels.append(pd.DataFrame({"id": raw.id, "label": int("fake" in entry["file"])}))
        raw = raw[["id", "news_url"]].copy()
        def host(value):
            try:
                domain = urlsplit(value if "://" in value else "https://"+value).hostname
                return (domain or "unknown").lower().removeprefix("www.")
            except ValueError:
                return "unknown"
        raw["publisher_domain"] = raw.news_url.map(host)
        raws.append(raw[["id", "publisher_domain"]])
    domains = pd.concat(raws).drop_duplicates("id")
    labels_table = pd.concat(raw_labels)
    id_labels = labels_table.groupby("id").label.nunique()
    conflicting_ids = set(id_labels[id_labels > 1].index)
    report["external_integrity_correction"] = {
        "conflicting_id_groups": len(conflicting_ids),
        "rule": "Remove IDs assigned both labels, in both model variants. Identified from raw integrity checks before inspecting external scores; no parameter/threshold changes.",
    }
    member_names = [member["name"] for member in bundle["consensus"].members]
    def refresh_metrics(predictions, variant, prefix):
        policy = report[variant]["policy"]
        member_probabilities = predictions[[f"member_{name}" for name in member_names]].to_numpy()
        metric = classification_with_interval(predictions.label, predictions.probability_fake, policy["threshold"])
        curve, ece = calibration_table(predictions.label, predictions.probability_fake)
        curve.to_csv(AUDIT_DIR / f"{prefix}_calibration.csv", index=False)
        metric["ece_10_bins"] = ece
        detail = {"member_probabilities": member_probabilities}
        agreement = agreement_report(predictions.label, detail, member_names, AUDIT_DIR, prefix)
        return {"metrics": metric, "selective_policy": policy_statistics(predictions.label, predictions.probability_fake, member_probabilities, policy), "agreement": agreement}
    for variant in ("before", "after"):
        predictions = pd.read_csv(AUDIT_DIR / f"{variant}_external_predictions.csv")
        predictions = predictions[~predictions.id.isin(conflicting_ids)].copy()
        predictions.to_csv(AUDIT_DIR / f"{variant}_external_predictions.csv", index=False)
        report[variant]["external_raw_headlines"] = refresh_metrics(predictions, variant, f"{variant}_external")
        if variant == "after":
            for source in report["external_by_source"]:
                source_path = AUDIT_DIR / f"after_external_{source}_predictions.csv"
                source_predictions = pd.read_csv(source_path)
                source_predictions = source_predictions[~source_predictions.id.isin(conflicting_ids)].copy()
                source_predictions.to_csv(source_path, index=False)
                report["external_by_source"][source] = refresh_metrics(source_predictions, "after", f"after_external_{source}")
        predictions = predictions.merge(domains, on="id", how="left", validate="many_to_one")
        predictions["publisher_domain"] = predictions.publisher_domain.fillna("unknown")
        counts = []
        rows = []
        for domain, group in predictions.groupby("publisher_domain"):
            y, pred = group.label.to_numpy(), group.prediction.to_numpy()
            confusion = np.asarray([((y==0)&(pred==0)).sum(), ((y==0)&(pred==1)).sum(), ((y==1)&(pred==0)).sum(), ((y==1)&(pred==1)).sum()], float)
            counts.append(confusion)
            rows.append({"publisher_domain": domain, "n": len(group), "macro_f1": macro_from_counts(confusion)})
        matrix = np.asarray(counts)
        rng = np.random.default_rng(42)
        scores = [
            macro_from_counts(matrix[rng.integers(0, len(matrix), len(matrix))].sum(axis=0))
            for _ in range(500)
        ]
        pd.DataFrame(rows).sort_values("n", ascending=False).to_csv(AUDIT_DIR / f"{variant}_publisher_groups.csv", index=False)
        metrics = report[variant]["external_raw_headlines"]["metrics"]
        metrics["publisher_cluster_macro_f1_ci95"] = np.quantile(scores, [.025, .975]).tolist()
        metrics["publisher_domains"] = len(counts)
        metrics["publisher_cluster_method"] = "500 domain-block bootstraps of already frozen predictions; unknown domains grouped together. Not a retuning or a new external test."
    external_manifest = report["external_manifest"]
    external_manifest["conflicting_id_groups"] = len(conflicting_ids)
    external_manifest["evaluated_rows"] = report["after"]["external_raw_headlines"]["metrics"]["rows"]
    write_json(AUDIT_DIR / "external_manifest.json", external_manifest)
    metrics = report["after"]["external_raw_headlines"]["metrics"]
    selective = report["after"]["external_raw_headlines"]["selective_policy"]
    report["release"]["checks"].update({
        "external_macro_f1": metrics["f1_macro"] >= .70,
        "external_macro_f1_ci_lower": metrics["macro_f1_ci95"][0] >= .65,
        "external_automatic_fake_precision": selective["fake_precision_wilson_lower"] >= .90,
        "external_automatic_real_precision": selective["real_precision_wilson_lower"] >= .90,
        "external_coverage": selective["coverage"] >= .50,
    })
    report["release"]["technical_gate_passed"] = all(value for key, value in report["release"]["checks"].items() if key != "redistribution_rights_confirmed")
    report["supplemental_diagnostics"] = {
        "trained_parameters_changed": False, "external_threshold_tuning": False,
        "description": "Train-only BH ANOVA q-values and publisher-domain uncertainty from frozen external scores.",
    }
    assert committee_fingerprint(bundle["consensus"]) == frozen["after_fingerprint"]
    bundle["metadata"]["external_metrics"] = report["after"]["external_raw_headlines"]["metrics"]
    bundle["metadata"]["release"] = report["release"]
    bundle["metadata"]["protocol_sha256"] = file_sha256(PROTOCOL_PATH)
    write_json(report_path, report)
    create_audit_markdown(report, AUDIT_DIR / "AUDIT.md")
    save_bundle(bundle, version_dir / "model.joblib")
    shutil.copy2(version_dir / "model.joblib", MODEL_PATH)
    sources = sorted((PROJECT_DIR / "news15").glob("*.py")) + sorted((PROJECT_DIR / "scripts").glob("*.py")) + sorted((PROJECT_DIR / "tests").glob("*.py"))
    sources += [PROJECT_DIR / "app.py", PROJECT_DIR / "README.md", PROJECT_DIR / "requirements.txt", PROTOCOL_PATH]
    manifest_sources = {}
    for source in sources:
        relative = source.relative_to(PROJECT_DIR)
        target = version_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest_sources[str(relative)] = file_sha256(source)
    report_snapshot = version_dir / "reports"
    shutil.copytree(AUDIT_DIR, report_snapshot)
    report_hashes = {str(path.relative_to(AUDIT_DIR)): file_sha256(path) for path in AUDIT_DIR.rglob("*") if path.is_file()}
    manifest = {
        "version": "2.0.0", "finalized": True, "model_sha256": file_sha256(version_dir / "model.joblib"),
        "protocol_sha256": file_sha256(PROTOCOL_PATH), "parameter_fingerprint": frozen["after_fingerprint"],
        "external_commit": report["external_manifest"]["commit"],
        "external_manifest_sha256": file_sha256(AUDIT_DIR / "external_manifest.json"),
        "report_sha256": file_sha256(report_path), "source_files": manifest_sources,
        "report_files": report_hashes,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "streamlit", "joblib")},
    }
    write_json(version_dir / "manifest.json", manifest)
    print(json.dumps({"version": "2.0.0", "external_macro_f1": bundle["metadata"]["external_metrics"]["f1_macro"], "source_files": len(manifest_sources), "classifier_and_policy_unchanged": True}, indent=2))


if __name__ == "__main__":
    finalize()

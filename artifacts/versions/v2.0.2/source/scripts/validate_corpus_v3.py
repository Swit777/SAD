"""Validate the candidate data release without fitting models or changing splits."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.corpus_expansion import MIN_BODY_WORDS, normalized_body
from news15.data import file_sha256
from news15.paths import PROJECT_DIR


def main():
    directory = PROJECT_DIR / "data" / "processed" / "corpus_v3"
    reports = PROJECT_DIR / "reports" / "corpus_expansion"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((reports / "import_report.json").read_text(encoding="utf-8"))
    checks = {}
    for name, digest in manifest["outputs"].items():
        checks[f"output_hash:{name}"] = file_sha256(directory / name) == digest
    for name, digest in manifest["implementation"].items():
        checks[f"implementation_hash:{name}"] = file_sha256(PROJECT_DIR / name) == digest
    checks["raw_welfake_hash"] = file_sha256(PROJECT_DIR / "data/raw/welfake/WELFake_Dataset.csv") == manifest["raw_welfake_sha256"]
    checks["base_ISOT_hash"] = file_sha256(PROJECT_DIR / "data/processed/articles.parquet") == manifest["input_ISOT_sha256"]
    protected = {
        "active_model": PROJECT_DIR / "artifacts/model.joblib",
        "original_ISOT_corpus": PROJECT_DIR / "data/processed/articles.parquet",
        "active_version_pointer": PROJECT_DIR / "artifacts/active_version.json",
    }
    checks["protected_artifacts_unchanged"] = all(file_sha256(path) == report["protected_hashes_before"][key] for key, path in protected.items())
    checks["training_remains_disabled"] = manifest["training_ready"] is False and report["frozen_model_retrained"] is False

    articles, hashes, groups = {}, set(), set()
    labels, eligibility = Counter(), Counter()
    body_valid = unique = reservations = new_metadata = unassigned = True
    for batch in pq.ParquetFile(directory / "articles.parquet").iter_batches(batch_size=1000):
        for row in batch.to_pylist():
            normalized = normalized_body(row["text"])
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            body_valid &= len(normalized) >= 100 and len(normalized.split()) >= MIN_BODY_WORDS and len(row["text"]) <= 500000
            unique &= row["id"] not in articles and row["group_id"] not in groups and digest not in hashes
            unique &= row["id"] == "corpus-v3-" + row["group_id"][:24]
            hashes.add(digest)
            groups.add(row["group_id"])
            unassigned &= row["split"] == "unassigned"
            splits = set(row["legacy_splits"])
            if "test" in splits:
                reservations &= row["eligibility"] in {"reserved_legacy_test", "reserved_external_seen"}
            elif "validation" in splits:
                reservations &= row["eligibility"] in {"reserved_legacy_test", "reserved_legacy_validation", "reserved_external_seen"}
            if row["eligibility"] == "new_candidate_unassigned":
                new_metadata &= not row["has_processed_ISOT"] and not row["raw_ISOT_overlap"] and not splits
                new_metadata &= row["published_at"] is None and row["original_publisher"] is None
            articles[row["id"]] = {key: row[key] for key in ("label", "eligibility", "provenance_rows")}
            labels[str(row["label"])] += 1
            eligibility[row["eligibility"]] += 1
    checks.update({
        "retained_count_matches": len(articles) == report["retained_unique_article_clusters"],
        "unique_ids_groups_and_normalized_bodies": bool(unique),
        "minimum_full_text_quality": bool(body_valid),
        "all_splits_unassigned": bool(unassigned),
        "legacy_evaluation_clusters_reserved": bool(reservations),
        "new_candidates_have_no_fabricated_metadata": bool(new_metadata),
        "label_counts_match": dict(labels) == report["retained_label_counts"] and set(labels) <= {"0", "1"},
        "eligibility_counts_match": dict(eligibility) == report["eligibility_counts"],
    })

    lineage = pd.read_parquet(directory / "provenance.parquet")
    counts = lineage.collection.value_counts().to_dict()
    n_welfake = report["WELFake_parsed_input_rows"]
    n_isot = report["ISOT_processed_input_rows"]
    checks["every_source_row_accounted_for"] = counts == {"WELFake": n_welfake, "ISOT": n_isot} and len(lineage) == n_welfake+n_isot
    checks["source_ids_unique_within_collection"] = not lineage.duplicated(["collection", "source_record_id"]).any()
    checks["welfake_row_ordinals_complete"] = set(lineage.loc[lineage.collection.eq("WELFake"), "source_record_id"].astype(int)) == set(range(n_welfake))
    quarantine = pd.read_csv(reports / "quarantine.csv")
    quarantine_ids = set(quarantine.id)
    checks["quarantine_separate_from_retained"] = quarantine_ids.isdisjoint(articles)
    checks["quarantine_count_matches"] = len(quarantine) == report["quarantined_clusters"]
    checks["quarantine_reasons_match"] = quarantine.reason.value_counts().to_dict() == report["quarantine_reasons"]
    checks["all_cluster_references_resolve"] = set(lineage.merged_id.dropna()) == set(articles) | quarantine_ids
    represented = Counter()
    lineage_consistent = True
    for row in lineage.itertuples():
        if row.merged_id in articles:
            article = articles[row.merged_id]
            lineage_consistent &= row.label == article["label"] and row.final_status == article["eligibility"]
            represented[row.merged_id] += 1
    checks["retained_labels_and_status_match_provenance"] = bool(lineage_consistent)
    checks["provenance_counts_match_each_article"] = all(represented[key] == article["provenance_rows"] for key, article in articles.items())
    edges = pd.read_csv(reports / "near_duplicate_edges.csv")
    checks["near_duplicate_matches_verified"] = len(edges) == report["processing_counts"]["near_duplicate_edges"] and bool(edges.word_5_shingle_jaccard.ge(.85).all())

    frozen = subprocess.run([sys.executable, str(PROJECT_DIR / "scripts/verify_version.py")], capture_output=True, text=True)
    checks["frozen_version_verification_passed"] = frozen.returncode == 0
    result = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks,
        "articles_checked": len(articles), "provenance_rows_checked": len(lineage),
        "limitation": "Artifact integrity and import invariants only; not independent factual verification or model validation.",
    }
    (reports / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if not result["passed"]:
        raise SystemExit("Candidate corpus validation failed.")


if __name__ == "__main__":
    main()

"""Check data integrity, label semantics, source accounting and held-out groups."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.data import file_sha256
from news15.multisource import OUTPUT, RAW, REPORTS, fingerprint, publisher_proxy
from news15.paths import PROJECT_DIR
from news15.release_checks import compatible_ui_pointer


def main():
    manifest = json.loads((OUTPUT/"manifest.json").read_text(encoding="utf-8"))
    report = json.loads((REPORTS/"import_report.json").read_text(encoding="utf-8"))
    checks = {}
    for section in ("source_hashes", "implementation"):
        checks[section+"_match"] = all(file_sha256(PROJECT_DIR/name) == digest for name, digest in manifest[section].items())
    if manifest.get("maintenance_implementation"):
        checks["maintenance_implementation_matches"] = all(file_sha256(PROJECT_DIR/name) == digest for name,digest in manifest["maintenance_implementation"].items())
    checks["output_hashes_match"] = all(file_sha256(OUTPUT/name) == digest for name, digest in manifest["outputs"].items())
    checks["report_hash_matches"] = file_sha256(REPORTS/"import_report.json") == manifest["report_sha256"]
    protected=report['protected_hashes']
    model_hash=next(v for k,v in protected.items() if k.replace('\\','/')=='artifacts/model.joblib')
    checks["earlier_releases_unchanged"] = all(
        compatible_ui_pointer(digest,model_hash) if name.replace('\\','/')=='artifacts/active_version.json'
        else file_sha256(PROJECT_DIR/name)==digest for name,digest in protected.items())
    checks["training_remains_disabled"] = manifest["training_ready"] is False and report["training_ready"] is False
    if report.get("recovery"):
        checkpoint = OUTPUT/"recovery_checkpoint"
        checks["recovery_inputs_match"] = all(file_sha256(checkpoint/name) == digest for name,digest in report["recovery"]["checkpoint"]["files"].items())
        previous = json.loads((checkpoint/"import_report.json").read_text(encoding="utf-8"))
        checks["recovered_new_texts_verified_against_originals"] = (
            report["recovery"]["verified_FakeNewsCorpus_texts"] == previous["new_source_counts"]["FakeNewsCorpus_retained"]
            and report["recovery"]["verified_FakeHealth_texts"] == previous["new_source_counts"]["FakeHealth_retained"]
            and report["recovery"]["scanned_prefix_records"] == previous["new_source_counts"]["FakeNewsCorpus_scanned"])
    old_hashes = set(pq.read_table(OUTPUT/"existing_body_hashes.parquet")["body_hash"].to_pylist())
    base = pd.read_parquet(PROJECT_DIR/"data/processed/corpus_v3/articles.parquet", columns=["id", "label", "eligibility"])
    old_rows = {r.id: (r.label, r.eligibility) for r in base.itertuples()}
    article_ids, body_hashes, source_ids, new_ids = set(), set(), set(), set()
    statuses, counts, domains, types = Counter(), Counter(), Counter(), Counter()
    unique = labels = dates = old_same = overlap_free = provenance_semantics = True
    sampled_texts = 0
    sampled_ok = True
    for batch in pq.ParquetFile(OUTPUT/"articles.parquet").iter_batches(batch_size=1000):
        for row in batch.to_pylist():
            identifier = row["id"]
            digest = row["body_hash"]
            unique &= identifier not in article_ids and digest not in body_hashes
            article_ids.add(identifier)
            body_hashes.add(digest)
            counts[row["collection"]] += 1
            provenance_semantics &= row["split"] == "unassigned"
            if identifier in old_rows:
                old_same &= (row["label"], row["eligibility"]) == old_rows[identifier]
                old_same &= row["label_scope"] == "collection_annotation"
            else:
                new_ids.add(identifier)
                source_key = (row["collection"], row["source_record_id"])
                unique &= source_key not in source_ids
                source_ids.add(source_key)
                overlap_free &= digest not in old_hashes
                labels &= row["label"] is None
                statuses[row["collection"]+":"+row["eligibility"]] += 1
                if row["collection"] == "FakeNewsCorpus":
                    labels &= row["label_scope"] == "publisher_proxy" and row["publisher_proxy_label"] == publisher_proxy(row["original_label"])
                    dates &= row["published_at"] is None and row["source_date_raw"] is None
                    domains[row["publisher"]] += 1
                    types[row["original_label"]] += 1
                else:
                    labels &= row["collection"] == "FakeHealth" and row["label_scope"] == "reporting_quality" and row["publisher_proxy_label"] is None
            # Deterministic sample of all articles, plus every imported health article.
            if row["collection"] == "FakeHealth" or int(digest[:8],16) % 100 == 0:
                sampled_texts += 1
                sampled_ok &= fingerprint(row["text"]) == digest and len(row["text"]) <= 500000
    checks.update({
        "article_count_matches": len(article_ids) == report["article_rows"],
        "new_article_count_matches": len(new_ids) == report["new_article_rows"],
        "article_ids_and_exact_body_hashes_unique": bool(unique),
        "all_base_ids_and_labels_preserved": set(old_rows).issubset(article_ids) and bool(old_same),
        "new_articles_do_not_overlap_prior_raw_bodies": bool(overlap_free),
        "new_article_labels_keep_their_actual_scope": bool(labels),
        "no_fabricated_FakeNewsCorpus_dates": bool(dates),
        "all_article_splits_unassigned": bool(provenance_semantics),
        "domain_count_matches": len(domains) == report["FakeNewsCorpus_domains"],
        "domain_cap_respected": max(domains.values(),default=0) <= report["selection"]["per_domain_cap"],
        "domain_and_type_counts_match": dict(domains) == report["FakeNewsCorpus_domain_counts"] and dict(types) == report["FakeNewsCorpus_type_counts"],
        "sampled_full_text_hashes_match": bool(sampled_ok),
    })
    lineage = pd.read_parquet(OUTPUT/"new_article_provenance.parquet")
    checks["new_article_source_ids_are_unique"] = not lineage.duplicated(["collection", "source_record_id"]).any()
    checks["every_scanned_article_accounted_for"] = all(int(lineage.collection.eq(source).sum()) == report["new_source_counts"].get(source+"_scanned",0) for source in ("FakeNewsCorpus", "FakeHealth"))
    checks["all_new_article_outputs_have_provenance"] = set(lineage.output_id.dropna()) == new_ids
    checks["article_decision_counts_match"] = Counter(lineage.collection+":"+lineage.status) == Counter(report["article_decisions"])

    # Compare emitted text to the raw claim column, never to the verdict/report column.
    expected = {}
    with zipfile.ZipFile(RAW/"liar/liar_dataset.zip") as archive:
        for split in ("train", "valid", "test"):
            with archive.open(split+".tsv") as raw:
                for row in csv.reader(io.TextIOWrapper(raw,encoding="utf-8",newline=""), delimiter="\t"):
                    if len(row) == 14 and row[2].strip():
                        expected[("LIAR",row[0])] = (row[2],row[1],split)
    with zipfile.ZipFile(RAW/"multifc/MultiFC_Codalab.zip") as archive:
        with zipfile.ZipFile(io.BytesIO(archive.read("public_data.zip"))) as public:
            for split in ("train", "dev", "test"):
                with public.open(split+".tsv") as raw:
                    for row in csv.reader(io.TextIOWrapper(raw,encoding="utf-8",newline=""),delimiter="\t"):
                        if len(row) == (12 if split == "test" else 13) and row[1].strip():
                            expected[("MultiFC",row[0])] = (row[1],None if split == "test" else row[2],split)
    claims = pd.read_parquet(OUTPUT/"claims.parquet")
    checks["no_verdict_report_or_speaker_history_feature_columns"] = not set(claims.columns) & {"reason","review","rating","speaker_history","article_title","checker"}
    checks["claim_source_ids_unique"] = not claims.duplicated(["collection","source_record_id"]).any()
    checks["claim_record_count_matches"] = len(claims) == report["claims"]["records"] == len(expected)
    raw_match = binary_ok = reserved = unassigned = True
    clusters = defaultdict(set)
    test_groups = set()
    validation_groups = set()
    unlabeled_test_count = 0
    for row in claims.itertuples():
        label = None if pd.isna(row.original_label) else row.original_label
        raw_match &= (row.text,label,row.original_split) == expected[(row.collection,row.source_record_id)]
        raw_match &= fingerprint(row.text) == row.group_id
        unassigned &= row.split == "unassigned"
        clusters[row.group_id].add(row.eligibility)
        if row.original_split == "test":
            test_groups.add(row.group_id)
            if row.collection == "MultiFC":
                unlabeled_test_count += 1
                binary_ok &= pd.isna(row.label) and label is None and row.label_scope == "unlabelled_claim"
        elif row.original_split in {"valid","dev"}:
            validation_groups.add(row.group_id)
        if not pd.isna(row.label):
            canonical = str(label).strip().lower()
            binary_ok &= canonical in {"true","false","pants-fire"} and row.label == (0 if canonical == "true" else 1)
    for group_id, statuses_in_group in clusters.items():
        reserved &= len(statuses_in_group) == 1
        if group_id in test_groups:
            reserved &= statuses_in_group.issubset({"reserved_original_test","reserved_external_seen","quarantine_conflicting_binary_labels"})
        elif group_id in validation_groups:
            reserved &= statuses_in_group.issubset({"reserved_original_validation","reserved_external_seen","quarantine_conflicting_binary_labels"})
    checks.update({
        "claim_text_and_original_labels_match_raw_columns": bool(raw_match),
        "MultiFC_test_labels_remain_missing": bool(binary_ok) and unlabeled_test_count > 0,
        "claim_groups_count_matches": len(clusters) == report["claims"]["exact_unique_claim_groups"],
        "claim_holdout_reservations_apply_to_whole_clusters": bool(reserved),
        "all_claim_splits_unassigned": bool(unassigned),
        "claim_collection_counts_match": claims.collection.value_counts().to_dict() == report["claims"]["collection_records"],
    })
    frozen = subprocess.run([sys.executable,str(PROJECT_DIR/"scripts/verify_version.py")],capture_output=True,text=True)
    checks["frozen_model_version_verified"] = frozen.returncode == 0
    result = {"validated_at_utc": datetime.now(timezone.utc).isoformat(), "passed": all(checks.values()),
        "checks": checks, "article_records_checked": len(article_ids), "sampled_full_texts_checked": sampled_texts,
        "claim_records_checked": len(claims), "validator_sha256": file_sha256(Path(__file__)),
        "limits": "Integrity, import semantics and exact overlap only. A UI-only successor is allowed only with archived parent pointer, identical model and unchanged scientific sources. Near-copy screening, sampling representativeness and factual correctness are not established."}
    (REPORTS/"validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=True,indent=2))
    if not result["passed"]:
        raise SystemExit("Corpus v4 validation failed.")


if __name__ == "__main__":
    main()

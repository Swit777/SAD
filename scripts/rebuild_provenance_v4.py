"""Independently replay source decisions and repair a corrupt provenance file."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from news15.data import file_sha256
from news15.corpus_expansion import normalized_body, write_json, headline_key
from news15.multisource import OUTPUT, REPORTS, RAW, LINEAGE_SCHEMA, BatchedWriter, fakecorpus_rows, reserved_headlines
from news15.paths import PROJECT_DIR


def main():
    report = json.loads((REPORTS/"import_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUTPUT/"manifest.json").read_text(encoding="utf-8"))
    for name,digest in report["source_hashes"].items():
        if file_sha256(PROJECT_DIR/name) != digest:
            raise RuntimeError("Raw source changed before provenance replay.")
    checkpoint = OUTPUT/"recovery_checkpoint"
    old = json.loads((checkpoint/"import_report.json").read_text(encoding="utf-8"))
    prefix = old["new_source_counts"]["FakeNewsCorpus_scanned"]
    old_hashes = set(pq.read_table(OUTPUT/"existing_body_hashes.parquet")["body_hash"].to_pylist())
    new_hashes = set()
    domains = Counter(old["FakeNewsCorpus_domain_counts"])
    counts = Counter(old["new_source_counts"])
    decisions = Counter(old["article_decisions"])
    kept = {}
    for batch in pq.ParquetFile(OUTPUT/"articles.parquet").iter_batches(batch_size=1000,columns=["id","collection","source_record_id","body_hash","title","text","original_label","publisher","source_url","eligibility"]):
        for row in batch.to_pylist():
            if row["collection"] == "FakeHealth":
                new_hashes.add(row["body_hash"])
            elif row["collection"] == "FakeNewsCorpus":
                kept[row["source_record_id"]] = {
                    "signature": (hashlib.sha256(row["text"].encode()).hexdigest(),row["title"],row["original_label"],row["publisher"],row["source_url"]),
                    "digest": row["body_hash"], "id": row["id"], "status": row["eligibility"]}
                if int(row["source_record_id"].split(":",1)[0]) < prefix:
                    new_hashes.add(row["body_hash"])
    temporary = OUTPUT/("provenance_rebuilt_"+uuid.uuid4().hex+".parquet")
    writer = BatchedWriter(temporary,LINEAGE_SCHEMA,batch_size=5000)
    prefix_ids = {}
    for batch in pq.ParquetFile(checkpoint/"new_article_provenance.parquet").iter_batches(batch_size=5000):
        for row in batch.to_pylist():
            writer.add(row)
            if row["collection"] == "FakeNewsCorpus":
                prefix_ids[int(row["source_record_id"].split(":",1)[0])] = row["source_record_id"]
    reserved = reserved_headlines()
    verified = set()
    limit = report["new_source_counts"]["FakeNewsCorpus_scanned"]
    for ordinal,raw in fakecorpus_rows(RAW/"fakenewscorpus/news.csv.z01"):
        if ordinal >= limit:
            break
        source_id = str(ordinal)+":"+str(raw["id"])
        selected = kept.get(source_id)
        if selected:
            actual = (hashlib.sha256(raw["content"].encode()).hexdigest(),raw["title"],raw["type"],raw["domain"],raw["url"])
            if actual != selected["signature"]:
                raise RuntimeError("Retained text or metadata does not match raw CSV.")
            verified.add(source_id)
        if ordinal < prefix:
            if prefix_ids.get(ordinal) != source_id:
                raise RuntimeError("Checkpoint prefix no longer aligns with CSV.")
            continue
        counts["FakeNewsCorpus_scanned"] += 1
        digest = identifier = None
        if "_import_error" in raw:
            status = raw["_import_error"]
        elif len(raw["content"]) > 500000 or "\ufffd" in raw["content"]:
            status = "invalid_or_oversized_body"
        elif domains[raw["domain"]] >= report["selection"]["per_domain_cap"]:
            status = "domain_cap"
        elif selected:
            digest = selected["digest"]
            if digest in old_hashes or digest in new_hashes:
                raise RuntimeError("A retained article is an exact duplicate.")
            status = "reserved_external_seen" if headline_key(raw["title"]) in reserved else "candidate_unassigned"
            if selected["status"] != status:
                raise RuntimeError("Retained reservation differs from raw title.")
            identifier = selected["id"]
            new_hashes.add(digest)
            domains[raw["domain"]] += 1
            counts["FakeNewsCorpus_retained"] += 1
        else:
            normalized = normalized_body(raw["content"])
            if len(normalized) < 100 or len(normalized.split()) < 20:
                status = "missing_or_short_body"
            else:
                digest = hashlib.sha256(normalized.encode()).hexdigest()
                if digest in old_hashes:
                    status = "existing_raw_corpus_overlap"
                elif digest in new_hashes:
                    status = "new_exact_duplicate"
                else:
                    raise RuntimeError(f"Unexplained omission at CSV ordinal {ordinal}.")
        if selected and identifier is None:
            raise RuntimeError("Source selection contradicts quality/domain limits.")
        decisions["FakeNewsCorpus:"+status] += 1
        writer.add({"collection":"FakeNewsCorpus","source_record_id":source_id,
                    "body_hash":digest,"output_id":identifier,"status":status})
        if (ordinal+1)%50000 == 0:
            print(f"Replayed {ordinal+1}/{limit} source records",flush=True)
    writer.close()
    if len(verified) != len(kept) or dict(counts) != report["new_source_counts"] or dict(decisions) != report["article_decisions"]:
        raise RuntimeError("Replayed decisions/counts differ from the import report.")
    actual_rows = sum(batch.num_rows for batch in pq.ParquetFile(temporary).iter_batches(batch_size=5000))
    if actual_rows != sum(counts[k+"_scanned"] for k in ("FakeHealth","FakeNewsCorpus")):
        raise RuntimeError("Rebuilt provenance is incomplete.")
    path = OUTPUT/"new_article_provenance.parquet"
    old_digest = file_sha256(path)
    backup = checkpoint/"corrupt_provenance.parquet"
    if not backup.exists():
        shutil.copy2(path,backup)
    temporary.replace(path)
    receipt = {"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "file":"new_article_provenance.parquet","old_sha256":old_digest,"new_sha256":file_sha256(path),
        "reason":"Parquet page-header integrity check failed; rebuilt in a unique temporary file from raw source decisions.",
        "raw_retained_texts_verified":len(verified),"source_rows_replayed":actual_rows,
        "all_decisions_and_counts_match":True,"readback_passed":True,
        "implementation_sha256":file_sha256(Path(__file__))}
    report["provenance_repair"] = receipt
    write_json(REPORTS/"import_report.json",report)
    manifest["outputs"]["new_article_provenance.parquet"] = receipt["new_sha256"]
    manifest["implementation"][str(Path(__file__).relative_to(PROJECT_DIR))] = receipt["implementation_sha256"]
    manifest["report_sha256"] = file_sha256(REPORTS/"import_report.json")
    manifest["provenance_repair"] = receipt
    write_json(OUTPUT/"manifest.json",manifest)
    write_json(REPORTS/"provenance_repair.json",receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == "__main__":
    main()

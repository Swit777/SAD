"""A separate multi-source data release; label scopes never silently collapse."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
from urllib.parse import urlsplit
import uuid
import zipfile
import zlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .corpus_expansion import headline_key, normalized_body, write_json
from .data import file_sha256
from .paths import PROJECT_DIR
from .data_run_lock import guard_data_run

RAW = PROJECT_DIR / "data/raw"
OUTPUT = PROJECT_DIR / "data/processed/corpus_v4"
REPORTS = PROJECT_DIR / "reports/multisource_expansion"
METHOD = "multisource-import-2"

ARTICLE_SCHEMA = pa.schema([
    ("id", pa.string()), ("group_id", pa.string()), ("body_hash", pa.string()),
    ("collection", pa.string()), ("source_record_id", pa.string()),
    ("title", pa.string()), ("text", pa.string()),
    ("label", pa.int8()), ("label_scope", pa.string()), ("original_label", pa.string()),
    ("publisher_proxy_label", pa.int8()), ("publisher", pa.string()),
    ("source_url", pa.string()), ("published_at", pa.timestamp("ms", tz="UTC")),
    ("source_date_raw", pa.string()), ("eligibility", pa.string()), ("split", pa.string()),
])
CLAIM_SCHEMA = pa.schema([
    ("id", pa.string()), ("group_id", pa.string()), ("collection", pa.string()),
    ("source_record_id", pa.string()), ("text", pa.string()), ("original_label", pa.string()),
    ("label", pa.int8()), ("label_scope", pa.string()), ("factcheck_source", pa.string()),
    ("source_url", pa.string()), ("source_date_raw", pa.string()),
    ("original_split", pa.string()), ("eligibility", pa.string()), ("split", pa.string()),
])
LINEAGE_SCHEMA = pa.schema([
    ("collection", pa.string()), ("source_record_id", pa.string()),
    ("body_hash", pa.string()), ("output_id", pa.string()), ("status", pa.string()),
])


class BatchedWriter:
    def __init__(self, path, schema, batch_size=1000):
        self.path = Path(path)
        self.temporary = self.path.with_name(self.path.stem+"."+uuid.uuid4().hex+".tmp.parquet")
        self.writer = pq.ParquetWriter(self.temporary, schema, compression="zstd")
        self.schema, self.batch_size = schema, batch_size
        self.pending = []
        self.count = 0

    def add(self, row):
        self.pending.append(row)
        self.count += 1
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.pending:
            self.writer.write_table(pa.Table.from_pylist(self.pending, schema=self.schema))
            self.pending.clear()

    def close(self):
        self.flush()
        self.writer.close()
        self.temporary.replace(self.path)


class PartialZipReader(io.RawIOBase):
    """Read the first DEFLATE member from the FIRST volume of a split ZIP.

    A split archive is intentionally incomplete: no full-member CRC is claimed.
    This reader only consumes the immutable local first volume, never code.
    """
    def __init__(self, stream):
        self.stream = stream
        signature = stream.read(4)
        if signature == b"PK\x07\x08":
            signature = stream.read(4)
        header = signature + stream.read(26)
        if len(header) != 30 or signature != b"PK\x03\x04":
            raise ValueError("Expected the first volume's ZIP local header.")
        fields = struct.unpack("<4s5H3I2H", header)
        if fields[2] & 1 or fields[3] != 8:
            raise ValueError("Only public, unencrypted DEFLATE CSV members are supported.")
        filename = stream.read(fields[-2]).decode("utf-8")
        stream.read(fields[-1])
        if filename != "news_cleaned_2018_02_13.csv":
            raise ValueError("Unexpected FakeNewsCorpus member name.")
        self.inflater = zlib.decompressobj(-15)
        self.pending = bytearray()
        self.finished = False

    def readable(self):
        return True

    def readinto(self, target):
        while not self.pending and not self.finished:
            compressed = self.stream.read(64*1024)
            if not compressed:
                self.finished = True
                break
            self.pending.extend(self.inflater.decompress(compressed))
            if self.inflater.eof:
                self.finished = True
        size = min(len(target), len(self.pending))
        target[:size] = self.pending[:size]
        del self.pending[:size]
        return size


def complete_lines(stream):
    """Discard a final physical fragment; strict CSV also rejects partial rows."""
    for line in stream:
        if not line.endswith(("\n", "\r")):
            break
        yield line


def fakecorpus_rows(path):
    csv.field_size_limit(32*1024*1024)
    with Path(path).open("rb") as raw:
        with io.TextIOWrapper(io.BufferedReader(PartialZipReader(raw)), encoding="utf-8", errors="replace", newline="") as text:
            reader = csv.DictReader(complete_lines(text), strict=True)
            expected = {"id", "domain", "type", "url", "content", "title", "source"}
            if not expected.issubset(reader.fieldnames or []):
                raise ValueError("Unknown FakeNewsCorpus CSV schema.")
            try:
                for ordinal, row in enumerate(reader):
                    if None in row or any(v is None for v in row.values()):
                        yield ordinal, {"id": row.get("id") or "", "_import_error": "invalid_csv_columns"}
                        continue
                    yield ordinal, row
            except csv.Error as exc:
                if "unexpected end of data" not in str(exc):
                    raise
                # The ZIP volume can end inside a quoted, multiline CSV field.
                return


def fingerprint(text):
    return hashlib.sha256(normalized_body(text).encode("utf-8")).hexdigest()


def publisher_proxy(value):
    # This is a publisher annotation, never an article's ground truth.
    return {"reliable": 0, "fake": 1}.get(value)


def claim_binary(value):
    # Mixed, partly true, numeric labels, promises and attribution are unresolved.
    return {"true": 0, "false": 1, "pants-fire": 1}.get(str(value).strip().lower())


def parse_multifc(row, split):
    labelled = split != "test"
    expected = 13 if labelled else 12
    if len(row) != expected:
        return None
    label = row[2] if labelled else None
    url_index = 3 if labelled else 2
    # The reason/report, speaker, checker and article title may reveal the verdict.
    # The only model-input text emitted is row[1], the original claim.
    return {
        "id": "multifc-"+row[0], "group_id": fingerprint(row[1]), "collection": "MultiFC",
        "source_record_id": row[0], "text": row[1], "original_label": label,
        "label": claim_binary(label) if labelled else None,
        "label_scope": "claim_factcheck" if labelled else "unlabelled_claim",
        "factcheck_source": row[0].split("-")[0], "source_url": row[url_index],
        "source_date_raw": row[10 if labelled else 9], "original_split": split,
        "eligibility": "unassigned", "split": "unassigned",
    }


def apply_claim_reservations(rows, reserved_titles=None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    conflict_groups = 0
    for group in grouped.values():
        labels = {r["label"] for r in group if r["label"] is not None}
        splits = {r["original_split"] for r in group}
        if len(labels) > 1:
            eligibility = "quarantine_conflicting_binary_labels"
            conflict_groups += 1
            for row in group:
                row["label"] = None
        elif reserved_titles and any(headline_key(r["text"]) in reserved_titles for r in group):
            eligibility = "reserved_external_seen"
        elif "test" in splits:
            eligibility = "reserved_original_test"
        elif splits & {"dev", "valid", "validation"}:
            eligibility = "reserved_original_validation"
        else:
            eligibility = "candidate_original_train"
        for row in group:
            row["eligibility"] = eligibility
    return {"records": len(rows), "exact_unique_claim_groups": len(grouped),
            "groups_with_multiple_records": sum(len(group)>1 for group in grouped.values()),
            "conflicting_binary_label_groups": conflict_groups,
            "record_eligibility": dict(Counter(r["eligibility"] for r in rows)),
            "raw_label_counts": {source: dict(Counter(str(r["original_label"]) for r in rows if r["collection"] == source)) for source in {r["collection"] for r in rows}}}


def existing_reference(output):
    inputs = [RAW/"isot/True.csv", RAW/"isot/Fake.csv", RAW/"welfake/WELFake_Dataset.csv"]
    signatures = {str(path.relative_to(PROJECT_DIR)): file_sha256(path) for path in inputs}
    signatures["normalizer"] = file_sha256(Path(__file__).with_name("data.py"))+file_sha256(Path(__file__).with_name("corpus_expansion.py"))
    metadata = output/"existing_reference.json"
    cached = output/"existing_body_hashes.parquet"
    if metadata.exists() and cached.exists():
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("inputs") == signatures and record.get("sha256") == file_sha256(cached):
            return set(pq.read_table(cached)["body_hash"].to_pylist())
    hashes = set()
    for path in inputs:
        n = 0
        for chunk in pd.read_csv(path, usecols=["text"], keep_default_na=False, dtype=str, chunksize=2000):
            for body in chunk.text:
                if body.strip():
                    hashes.add(fingerprint(body))
            n += len(chunk)
            if n % 20000 == 0:
                print(f"Reference fingerprints: {path.name}: {n}", flush=True)
    pq.write_table(pa.table({"body_hash": sorted(hashes)}), cached, compression="zstd")
    write_json(metadata, {"inputs": signatures, "sha256": file_sha256(cached), "unique_body_hashes": len(hashes)})
    return hashes


def reserved_headlines():
    titles = set()
    for batch in pq.ParquetFile(PROJECT_DIR/"data/processed/corpus_v3/articles.parquet").iter_batches(columns=["title", "eligibility"]):
        for row in batch.to_pylist():
            if row["eligibility"].startswith("reserved_") and headline_key(row["title"]):
                titles.add(headline_key(row["title"]))
    for path in (RAW/"fakenewsnet").glob("*.csv"):
        titles.update(headline_key(t) for t in pd.read_csv(path, usecols=["title"], keep_default_na=False).title if headline_key(t))
    return titles


def ensure_sources():
    specs = {
        "liar": ("liar_dataset.zip", "https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip", 1013571),
        "multifc": ("MultiFC_Codalab.zip", "https://copenlu.github.io/publication/2019_emnlp_augenstein/MultiFC_Codalab.zip", 33563576),
        "fakehealth": ("FakeHealth.zip", "https://codeload.github.com/EnyanDai/FakeHealth/zip/ec9379de8f8f13af8c436dd6dd9bfaddacd2df30", 16784362),
        "fakenewscorpus": ("news.csv.z01", "https://github.com/several27/FakeNewsCorpus/releases/download/v1.0/news.csv.z01", 1073741824),
    }
    hashes = {}
    for source, (filename, url, size) in specs.items():
        directory = RAW/source
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/filename
        if not path.exists():
            print(f"Downloading {source}...", flush=True)
            partial = path.with_suffix(path.suffix+".part")
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with partial.open("wb") as target:
                    for block in response.iter_content(1024*1024):
                        target.write(block)
            if partial.stat().st_size != size:
                raise ValueError(f"Unexpected download size: {source}")
            partial.replace(path)
        if path.stat().st_size != size:
            raise ValueError(f"Source size mismatch: {source}")
        if filename.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                if archive.testzip():
                    raise ValueError(f"ZIP CRC failure: {source}")
        digest = file_sha256(path)
        source_manifest = directory/"source.json"
        if source_manifest.exists():
            old = json.loads(source_manifest.read_text(encoding="utf-8"))
            if old.get("sha256") != digest:
                raise ValueError(f"Previously recorded source checksum mismatch: {source}")
        else:
            write_json(source_manifest, {"dataset": source, "url": url, "sha256": digest, "bytes": size,
                                        "full_member_crc_verified": filename.endswith(".zip")})
        hashes[str(path.relative_to(PROJECT_DIR))] = digest
    return hashes


def collect_claims(reserved):
    rows, rejected = [], Counter()
    with zipfile.ZipFile(RAW/"liar/liar_dataset.zip") as archive:
        for split in ("train", "valid", "test"):
            reader = csv.reader(io.StringIO(archive.read(split+".tsv").decode("utf-8")), delimiter="\t")
            for row in reader:
                if len(row) != 14 or not row[2].strip():
                    rejected["LIAR_bad_columns_or_empty_claim"] += 1
                    continue
                rows.append({"id": "liar-"+row[0], "group_id": fingerprint(row[2]), "collection": "LIAR",
                    "source_record_id": row[0], "text": row[2], "original_label": row[1], "label": claim_binary(row[1]),
                    "label_scope": "claim_factcheck", "factcheck_source": "PolitiFact", "source_url": None,
                    "source_date_raw": None, "original_split": split, "eligibility": "unassigned", "split": "unassigned"})
    with zipfile.ZipFile(RAW/"multifc/MultiFC_Codalab.zip") as archive:
        with zipfile.ZipFile(io.BytesIO(archive.read("public_data.zip"))) as public:
            if public.testzip():
                raise ValueError("Nested MultiFC ZIP CRC failed.")
            for split in ("train", "dev", "test"):
                reader = csv.reader(io.StringIO(public.read(split+".tsv").decode("utf-8")), delimiter="\t")
                for raw in reader:
                    row = parse_multifc(raw, split)
                    if row is None or not row["text"].strip():
                        rejected["MultiFC_bad_columns_or_empty_claim"] += 1
                    else:
                        rows.append(row)
    summary = apply_claim_reservations(rows, reserved)
    summary["rejected_rows"] = dict(rejected)
    summary["collection_records"] = dict(Counter(row["collection"] for row in rows))
    return rows, summary


def recovery_checkpoint():
    directory = OUTPUT/"recovery_checkpoint"
    directory.mkdir(exist_ok=True)
    record_path = directory/"checkpoint.json"
    if not record_path.exists():
        files = {"articles.parquet": OUTPUT/"articles.parquet",
                 "new_article_provenance.parquet": OUTPUT/"new_article_provenance.parquet",
                 "import_report.json": REPORTS/"import_report.json"}
        record = {"stage": "completed files from an interrupted, unfinalized import", "files": {}}
        for name, path in files.items():
            digest = file_sha256(path)
            shutil.copy2(path, directory/name)
            if file_sha256(directory/name) != digest:
                raise RuntimeError("Checkpoint copy verification failed.")
            record["files"][name] = digest
        write_json(record_path, record)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if any(file_sha256(directory/name) != digest for name,digest in record["files"].items()):
        raise RuntimeError("Recovery checkpoint changed.")
    old = json.loads((directory/"import_report.json").read_text(encoding="utf-8"))
    for name,digest in {**old["source_hashes"], **old["protected_hashes"]}.items():
        if file_sha256(PROJECT_DIR/name) != digest:
            raise RuntimeError("Recovery inputs no longer match.")
    if pq.ParquetFile(directory/"articles.parquet").metadata.num_rows != old["article_rows"]:
        raise RuntimeError("Recovery article count differs from its report.")
    return directory, old, record


@guard_data_run(OUTPUT/".import.lock")
def import_sources(target_articles=250000, domain_cap=10000, resume=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if target_articles <= 0 or domain_cap <= 0:
        raise ValueError("Selection limits must be positive.")
    protected = [PROJECT_DIR/"artifacts/model.joblib", PROJECT_DIR/"artifacts/active_version.json",
                 PROJECT_DIR/"data/processed/articles.parquet", PROJECT_DIR/"data/processed/corpus_v3/articles.parquet",
                 PROJECT_DIR/"data/processed/corpus_v3/manifest.json"]
    before = {path.relative_to(PROJECT_DIR).as_posix(): file_sha256(path) for path in protected}
    source_hashes = ensure_sources()
    seen = existing_reference(OUTPUT)
    reserved = reserved_headlines()
    writer = BatchedWriter(OUTPUT/"articles.parquet", ARTICLE_SCHEMA)
    lineage = BatchedWriter(OUTPUT/"new_article_provenance.parquet", LINEAGE_SCHEMA, batch_size=5000)
    counts, statuses, domains, types = Counter(), Counter(), Counter(), Counter()
    base_count = 0
    new_hashes = set()
    prefix_expected, prefix_verified, prefix_ids = {}, set(), {}
    health_expected, health_verified = {}, set()
    resume_skip = 0
    resume_record = None
    if resume:
        directory, old, resume_record = recovery_checkpoint()
        if old["selection"]["per_domain_cap"] != domain_cap:
            raise ValueError("A checkpoint must retain its original domain cap.")
        print("Recovering completed records and verifying them against original texts...", flush=True)
        counts.update(old["new_source_counts"])
        statuses.update(old["article_decisions"])
        domains.update(old["FakeNewsCorpus_domain_counts"])
        types.update(old["FakeNewsCorpus_type_counts"])
        resume_skip = counts["FakeNewsCorpus_scanned"]
        # Fast raw-text digests verify cache identity; the normalizer is unchanged.
        originals = {}
        for batch in pq.ParquetFile(PROJECT_DIR/"data/processed/corpus_v3/articles.parquet").iter_batches(batch_size=1000, columns=["id","text","title","label","eligibility"]):
            for row in batch.to_pylist():
                originals[row["id"]] = (hashlib.sha256(row["text"].encode()).hexdigest(),row["title"],row["label"],row["eligibility"])
        for batch in pq.ParquetFile(directory/"articles.parquet").iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                digest = hashlib.sha256(row["text"].encode()).hexdigest()
                if row["collection"] not in {"FakeNewsCorpus","FakeHealth"}:
                    if originals.pop(row["id"],None) != (digest,row["title"],row["label"],row["eligibility"]):
                        raise RuntimeError("Recovered base text or label differs from v3.")
                    base_count += 1
                else:
                    new_hashes.add(row["body_hash"])
                    if row["collection"] == "FakeNewsCorpus":
                        prefix_expected[row["source_record_id"]] = (digest,row["title"],row["original_label"],row["publisher"],row["source_url"])
                    else:
                        health_expected[row["source_record_id"]] = (digest,row["original_label"])
                writer.add(row)
        if originals or base_count != old["base_v3_articles"]:
            raise RuntimeError("Recovery did not preserve the complete v3 base.")
        for batch in pq.ParquetFile(directory/"new_article_provenance.parquet").iter_batches(batch_size=5000):
            for row in batch.to_pylist():
                lineage.add(row)
                if row["collection"] == "FakeNewsCorpus":
                    ordinal = int(row["source_record_id"].split(":",1)[0])
                    prefix_ids[ordinal] = row["source_record_id"]
        if set(prefix_ids) != set(range(resume_skip)):
            raise RuntimeError("Recovery source prefix is incomplete.")
    for batch in (() if resume else pq.ParquetFile(PROJECT_DIR/"data/processed/corpus_v3/articles.parquet").iter_batches(batch_size=1000)):
        for row in batch.to_pylist():
            base_count += 1
            writer.add({"id": row["id"], "group_id": row["group_id"], "body_hash": fingerprint(row["text"]),
                "collection": "+".join(row["collections"]), "source_record_id": row["id"],
                "title": row["title"], "text": row["text"], "label": row["label"],
                "label_scope": "collection_annotation", "original_label": str(row["label"]),
                "publisher_proxy_label": None, "publisher": None, "source_url": None,
                "published_at": pd.Timestamp(row["published_at"]).tz_localize("UTC").to_pydatetime() if row["published_at"] else None,
                "source_date_raw": str(row["published_at"]) if row["published_at"] else None,
                "eligibility": row["eligibility"], "split": "unassigned"})

    def add_article(collection, source_id, title, body, original_label, publisher=None, source_url=None, date=None):
        counts[collection+"_scanned"] += 1
        status, digest, identifier = None, None, None
        if not isinstance(body, str) or len(body) > 500000 or "\ufffd" in body:
            status = "invalid_or_oversized_body"
        elif collection == "FakeNewsCorpus" and domains[publisher] >= domain_cap:
            # Rejected by quota regardless of text; avoid expensive normalization.
            status = "domain_cap"
        else:
            normalized = normalized_body(body)
            if len(normalized) < 100 or len(normalized.split()) < 20:
                status = "missing_or_short_body"
            else:
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                if digest in seen:
                    status = "existing_raw_corpus_overlap"
                elif digest in new_hashes:
                    status = "new_exact_duplicate"
                elif collection == "FakeNewsCorpus" and domains[publisher] >= domain_cap:
                    status = "domain_cap"
        if status is None:
            new_hashes.add(digest)
            identifier = "corpus-v4-"+digest[:24]
            status = "reserved_external_seen" if headline_key(title) in reserved else "candidate_unassigned"
            timestamp = None
            if collection == "FakeHealth" and isinstance(date, (int, float)) and 0 < date < 4102444800:
                timestamp = datetime.fromtimestamp(date, tz=timezone.utc)
            writer.add({"id": identifier, "group_id": digest, "body_hash": digest, "collection": collection,
                "source_record_id": str(source_id), "title": title or "", "text": body,
                "label": None, "label_scope": "publisher_proxy" if collection == "FakeNewsCorpus" else "reporting_quality",
                "original_label": str(original_label) if original_label is not None else None,
                "publisher_proxy_label": publisher_proxy(original_label) if collection == "FakeNewsCorpus" else None,
                "publisher": publisher or None, "source_url": source_url or None,
                "published_at": timestamp, "source_date_raw": str(date) if date is not None else None,
                "eligibility": status, "split": "unassigned"})
            counts[collection+"_retained"] += 1
            if collection == "FakeNewsCorpus":
                domains[publisher] += 1
                types[str(original_label)] += 1
        statuses[collection+":"+status] += 1
        lineage.add({"collection": collection, "source_record_id": str(source_id), "body_hash": digest,
                     "output_id": identifier, "status": status})

    print("Importing FakeHealth full texts, excluding review verdicts...", flush=True)
    with zipfile.ZipFile(RAW/"fakehealth/FakeHealth.zip") as archive:
        names = archive.namelist()
        prefix = names[0]
        for subset in ("HealthRelease", "HealthStory"):
            reviews = {r["news_id"]: r for r in json.loads(archive.read(prefix+f"dataset/reviews/{subset}.json"))}
            for name in sorted(n for n in names if f"/content/{subset}/" in n and n.endswith(".json")):
                content = json.loads(archive.read(name))
                source_id = Path(name).stem
                review = reviews.get(source_id, {})
                if resume:
                    key = subset+"/"+source_id
                    if key in health_expected:
                        rating = review.get("rating")
                        current = (hashlib.sha256(content.get("text","").encode()).hexdigest(),None if rating is None else str(rating))
                        if current != health_expected[key]:
                            raise RuntimeError("Recovered health text/rating differs from source.")
                        health_verified.add(key)
                    continue
                url = review.get("source_link") or content.get("url")
                host = urlsplit(url).hostname if url else None
                add_article("FakeHealth", subset+"/"+source_id, content.get("title"), content.get("text"),
                    review.get("rating"), host, url, content.get("publish_date"))
    print("Streaming the public FakeNewsCorpus first ZIP volume...", flush=True)
    stopped_at_target = False
    for ordinal, raw in fakecorpus_rows(RAW/"fakenewscorpus/news.csv.z01"):
        source_id = str(ordinal)+":"+str(raw["id"])
        if ordinal < resume_skip:
            if prefix_ids[ordinal] != source_id:
                raise RuntimeError("Recovered prefix record IDs no longer align.")
            if source_id in prefix_expected:
                current = (hashlib.sha256(raw["content"].encode()).hexdigest(),raw["title"],raw["type"],raw["domain"],raw["url"])
                if current != prefix_expected[source_id]:
                    raise RuntimeError("Recovered article differs from the original CSV.")
                prefix_verified.add(source_id)
            continue
        if ordinal == resume_skip and resume:
            if len(prefix_verified) != len(prefix_expected) or len(health_verified) != len(health_expected):
                raise RuntimeError("Recovery source verification incomplete.")
            print(f"Verified recovered prefix of {resume_skip} CSV records; continuing...", flush=True)
        if "_import_error" in raw:
            counts["FakeNewsCorpus_scanned"] += 1
            status = raw["_import_error"]
            statuses["FakeNewsCorpus:"+status] += 1
            lineage.add({"collection": "FakeNewsCorpus", "source_record_id": source_id,
                         "body_hash": None, "output_id": None, "status": status})
        else:
            add_article("FakeNewsCorpus", source_id, raw["title"], raw["content"], raw["type"], raw["domain"], raw["url"])
        if (ordinal+1) % 10000 == 0:
            print(f"FakeNewsCorpus scanned={ordinal+1}, retained={counts['FakeNewsCorpus_retained']}, domains={len(domains)}", flush=True)
        if counts["FakeNewsCorpus_retained"] >= target_articles:
            stopped_at_target = True
            break
    writer.close()
    lineage.close()
    print("Importing human fact-checked claims, preserving original splits...", flush=True)
    claims, claim_report = collect_claims(reserved)
    claim_writer = BatchedWriter(OUTPUT/"claims.parquet", CLAIM_SCHEMA)
    for row in claims:
        claim_writer.add(row)
    claim_writer.close()
    report = {
        "version": "corpus-v4-candidate", "method": METHOD, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_v3_articles": base_count, "article_rows": writer.count, "new_article_rows": writer.count-base_count,
        "new_source_counts": dict(counts), "article_decisions": dict(statuses),
        "FakeNewsCorpus_domains": len(domains), "FakeNewsCorpus_domain_counts": dict(domains),
        "FakeNewsCorpus_type_counts": dict(types), "claims": claim_report,
        "selection": {"requested_FakeNewsCorpus_retained": target_articles, "per_domain_cap": domain_cap,
                      "stopped_at_target": stopped_at_target, "scope": "Sequential prefix of the first split-ZIP volume. Not a uniform sample of the full corpus."},
        "deduplication": "Exact normalized body fingerprints against ALL raw ISOT and WELFake, then within new articles. Claims use exact normalized text clusters across LIAR and MultiFC. Semantic/near-duplicate screening of new records is pending.",
        "label_policy": "New article label is null: publisher reliability and reporting quality are separate tasks. Claim labels retain their original scales; only literal true/false/pants-fire have a conservative binary view; no majority vote.",
        "test_policy": "All splits unassigned. Original claim validation/test clusters and previously seen external titles reserved. No model fitting or threshold selection.",
        "rights": "Local research only. FakeNewsCorpus repository Apache-2.0 does not establish third-party article rights. LIAR explicitly research-only; MultiFC and FakeHealth do not provide a clear blanket data license in downloaded materials. No public redistribution.",
        "source_hashes": source_hashes, "protected_hashes": before,
        "recovery": None if not resume else {"checkpoint": resume_record, "verified_FakeNewsCorpus_texts": len(prefix_verified), "verified_FakeHealth_texts": len(health_verified), "scanned_prefix_records": resume_skip},
        "quota_policy": "Method 2 checks exhausted domain quotas before normalization; the recovered prefix retains its earlier decision reasons. Retained-article selection is unchanged.",
        "training_ready": False,
    }
    if any(file_sha256(PROJECT_DIR/path) != digest for path, digest in before.items()):
        raise RuntimeError("An earlier release or frozen artifact changed.")
    report["protected_artifacts_unchanged"] = True
    write_json(REPORTS/"import_report.json", report)
    outputs = {name: file_sha256(OUTPUT/name) for name in ("articles.parquet", "claims.parquet", "new_article_provenance.parquet", "existing_body_hashes.parquet", "existing_reference.json")}
    implementation = {str(path.relative_to(PROJECT_DIR)): file_sha256(path) for path in
                      [Path(__file__), Path(__file__).with_name("corpus_expansion.py"), Path(__file__).with_name("data.py"), PROJECT_DIR/"scripts/expand_sources.py"]}
    write_json(OUTPUT/"manifest.json", {"version": report["version"], "method": METHOD, "source_hashes": source_hashes,
        "base_v3_sha256": before["data/processed/corpus_v3/articles.parquet"], "outputs": outputs,
        "implementation": implementation, "report_sha256": file_sha256(REPORTS/"import_report.json"), "training_ready": False})
    write_report(report)
    print(json.dumps({key: report[key] for key in ("article_rows", "new_article_rows", "new_source_counts", "FakeNewsCorpus_domains", "protected_artifacts_unchanged")}, indent=2), flush=True)
    print(json.dumps({key: claim_report[key] for key in ("records", "exact_unique_claim_groups", "collection_records", "rejected_rows")}, indent=2), flush=True)
    return report


def write_report(report):
    c = report["new_source_counts"]
    claims = report["claims"]
    lines = ["# Расширение данных: корпус v4", "",
        f"Полных текстов статей: {report['article_rows']:,}; новых относительно прежних raw ISOT/WELFake: {report['new_article_rows']:,}.",
        f"FakeNewsCorpus: {c.get('FakeNewsCorpus_retained',0):,} сохранено из {c.get('FakeNewsCorpus_scanned',0):,} просмотренных записей, {report['FakeNewsCorpus_domains']} доменов.",
        f"FakeHealth: {c.get('FakeHealth_retained',0):,} сохранено из {c.get('FakeHealth_scanned',0):,} файлов полных текстов.",
        f"LIAR + MultiFC: {claims['records']:,} записей утверждений, {claims['exact_unique_claim_groups']:,} точных групп после нормализации.",
        "", "## Что означают эти числа", "",
        "Статьи и короткие утверждения находятся в разных таблицах и не суммируются в число уникальных новостей.",
        "У новых статей label=NULL: FakeNewsCorpus размечен по сайтам, FakeHealth — по качеству освещения. Это не проверка истинности каждой статьи.",
        "Для FakeNewsCorpus сохранены исходные категории и отдельный publisher_proxy_label только для fake/reliable. Остальные категории не объявлены истинными или ложными автоматически.",
        "У LIAR и MultiFC сохранены исходные шкалы меток. Числовые и смешанные категории не обращены в бинарные по догадке.",
        "Тест MultiFC не содержит меток; они оставлены пустыми. Проверочные кластеры зарезервированы целиком, в том числе при совпадении утверждений между коллекциями.",
        "Тексты разборов, рейтинги, вердикты и кредитная история спикера не добавлены к проверяемому тексту.",
        "", "## Ограничения", "",
        "Загружена первая часть публичного многотомного ZIP FakeNewsCorpus (1 ГиБ), а не все 9,4 млн статей. Выборка является последовательным префиксом с ограничением на домен; её репрезентативность всему набору не установлена.",
        "SHA-256 локальной части сохранён. Полная CRC-проверка сжатого многотомного CSV невозможна без остальных частей; для трёх небольших ZIP проверены CRC, включая вложенный ZIP MultiFC.",
        "Проверены точные совпадения со всеми raw ISOT и WELFake. Поиск близких копий и семантических пересказов в новых данных остаётся этапом подготовки нового обучения.",
        "Исходные даты отсутствуют у FakeNewsCorpus: scraped_at не выдан за published_at. Даты из FakeHealth сохранены как полученные метаданные, без независимой проверки.",
        "Все split=unassigned, training_ready=false. Прежние версии и действующая модель сохранены. Обучения, настройки порога и новой оценки качества не было.",
        "", "## Файлы", "",
        "- data/processed/corpus_v4/articles.parquet — общий реестр полных текстов.",
        "- data/processed/corpus_v4/claims.parquet — утверждения и исходные проверочные группы.",
        "- data/processed/corpus_v4/new_article_provenance.parquet — решение для каждой просмотренной новой статьи.",
        "- data/processed/corpus_v4/manifest.json — входы, выходы и реализация с SHA-256.",
        "- reports/multisource_expansion/import_report.json — подробные численные результаты.",
        "- reports/multisource_expansion/SOURCES.md — источники и условия.",
    ]
    (REPORTS/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

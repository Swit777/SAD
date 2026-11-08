"""Import WELFake as a separate, provenance-preserving candidate data release."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .data import clean_text, file_sha256, load_articles
from .paths import PROJECT_DIR, RAW_DIR, REPORTS_DIR

WELFAKE_URL = "https://zenodo.org/records/4561253/files/WELFake_Dataset.csv?download=1"
WELFAKE_MD5 = "73c9675a4b3d09f86a6933d0b8d7d908"
WELFAKE_BYTES = 245086152
RAW_WELFAKE = PROJECT_DIR / "data" / "raw" / "welfake"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "corpus_v3"
REPORT_DIR = REPORTS_DIR / "corpus_expansion"
METHOD_VERSION = "welfake-import-1"
MIN_BODY_WORDS = 20


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def download_welfake(raw_dir=RAW_WELFAKE):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "WELFake_Dataset.csv"
    if not path.exists():
        partial = raw_dir / "WELFake_Dataset.csv.part"
        digest = hashlib.md5()
        received = 0
        print("Downloading original WELFake CSV (245 MB)...", flush=True)
        with requests.get(WELFAKE_URL, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024*1024):
                    if chunk:
                        stream.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                        if received // (32*1024*1024) != (received-len(chunk)) // (32*1024*1024):
                            print(f"Downloaded {received//(1024*1024)} MiB", flush=True)
        if received != WELFAKE_BYTES or digest.hexdigest() != WELFAKE_MD5:
            raise ValueError("Original WELFake checksum/size mismatch; partial file retained for inspection.")
        partial.replace(path)
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    if path.stat().st_size != WELFAKE_BYTES or digest.hexdigest() != WELFAKE_MD5:
        raise ValueError("Cached WELFake file does not match the published original.")
    manifest = {
        "dataset": "WELFake", "record": "https://zenodo.org/records/4561253",
        "url": WELFAKE_URL, "bytes": path.stat().st_size,
        "published_md5": WELFAKE_MD5, "actual_md5": digest.hexdigest(),
        "sha256": file_sha256(path), "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_labels": {"0": "fake", "1": "real"},
        "project_labels": {"0": "real", "1": "fake"},
        "documented_mapping": "project_label = 1 - WELFake_label",
        "mapping_status": "Author documentation only; import verifies the actual file against training copies.",
        "upstream_collections": ["Kaggle", "McIntire", "Reuters", "BuzzFeed Political"],
        "row_level_original_publisher": "not provided in this CSV",
        "rights": "Public download verified; redistribution clearance is not established by this import.",
    }
    write_json(raw_dir / "source.json", manifest)
    return path


def normalized_body(text):
    return " ".join(re.findall(r"\w+", clean_text(text), flags=re.UNICODE))


def body_hash(text):
    return hashlib.sha256(normalized_body(text).encode("utf-8")).hexdigest()


def headline_key(title):
    value = clean_text(title)
    value = re.sub(r"^(?:update\s*\d*|corrected|exclusive|breaking)\s*[-:–—]*\s*", "", value)
    return " ".join(re.findall(r"\w+", value))


def normalize_label(value, convention="documented_inverse"):
    value = str(value).strip()
    if value not in ("0", "1"):
        raise ValueError(f"Invalid WELFake label {value!r}; no guessing or silent coercion.")
    if convention == "documented_inverse":
        return 1-int(value)
    if convention == "verified_identity":
        return int(value)
    raise ValueError("Unknown label convention.")


def audit_label_mapping(csv_path, training_labels, min_evidence=1000):
    """Resolve coding semantics using copied TRAIN articles, never model scores."""
    counts = Counter()
    seen = set()
    examples = []
    parsed_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=["text", "label"], keep_default_na=False, dtype=str, chunksize=2000):
        for text, raw_label in zip(chunk.text, chunk.label):
            parsed_rows += 1
            try:
                raw_value = int(str(raw_label).strip())
                if str(raw_label).strip() not in ("0", "1"):
                    raise ValueError
            except ValueError:
                counts["invalid_labels"] += 1
                continue
            normalized = normalized_body(text)
            if not normalized:
                continue
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            if digest in seen or training_labels.get(digest) not in (1, 2):
                continue
            seen.add(digest)
            target = 0 if training_labels[digest] == 1 else 1
            counts["unique_training_copies"] += 1
            counts["identity_agreements"] += int(raw_value == target)
            counts["documented_inverse_agreements"] += int(1-raw_value == target)
            counts[f"raw_{raw_value}_project_{target}"] += 1
            if len(examples) < 12:
                examples.append({"row_index": parsed_rows-1, "body_hash": digest, "raw_label": raw_value, "existing_training_label": target})
    n = counts["unique_training_copies"]
    identity_rate = counts["identity_agreements"]/n if n else 0
    inverse_rate = counts["documented_inverse_agreements"]/n if n else 0
    if n < min_evidence or max(identity_rate, inverse_rate) < .99:
        raise ValueError(f"Ambiguous label encoding: {n} unique TRAIN matches; identity={identity_rate:.4f}, documented inverse={inverse_rate:.4f}. Import stopped.")
    convention = "verified_identity" if identity_rate > inverse_rate else "documented_inverse"
    return {
        "parsed_rows": parsed_rows, "evidence": dict(counts), "examples": examples,
        "identity_agreement_rate": identity_rate, "documented_inverse_agreement_rate": inverse_rate,
        "effective_convention": convention,
        "documentation_discrepancy": convention == "verified_identity",
        "basis": "Encoding reconciliation from identical articles already labelled in ISOT TRAIN; no predictions, validation/test labels, or model-score maximization.",
        "limitation": "Identical-content agreement resolves code orientation for this artifact; it does not independently verify factual truth.",
    }


def raw_isot_reference(raw_dir=RAW_DIR):
    """Fingerprint all raw ISOT rows, not only the previously dated subset."""
    reference = {}
    for name, label in (("True.csv", 0), ("Fake.csv", 1)):
        path = Path(raw_dir) / name
        if not path.exists():
            raise FileNotFoundError("Original ISOT files are required to audit cross-corpus overlap.")
        for chunk in pd.read_csv(path, keep_default_na=False, chunksize=2000, usecols=["text"]):
            for text in chunk.text:
                normalized = normalized_body(text)
                if normalized:
                    digest = hashlib.sha256(normalized.encode()).hexdigest()
                    reference[digest] = reference.get(digest, 0) | (1 << label)
    return reference


@lru_cache(maxsize=100000)
def _token_hash(word):
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")


def simhash64(text):
    """Candidate generation only; every removal also needs shingle verification."""
    counts = Counter(word for word in normalized_body(text).split() if word not in ENGLISH_STOP_WORDS and len(word) >= 3)
    frequent = counts.most_common(96)
    if not frequent:
        return 0
    hashes = np.asarray([_token_hash(word) for word, _ in frequent], dtype=np.uint64)
    weights = np.asarray([min(count, 5) for _, count in frequent], dtype=np.int16)
    positions = np.arange(64, dtype=np.uint64)
    signs = ((((hashes[:, None] >> positions) & 1).astype(np.int16)*2-1)*weights[:, None]).sum(axis=0)
    result = 0
    for position in np.flatnonzero(signs >= 0):
        result |= 1 << int(position)
    return result


def word_shingles(text, width=5):
    words = normalized_body(text).split()
    return frozenset(" ".join(words[i:i+width]) for i in range(max(0, len(words)-width+1)))


class CorpusBuilder:
    """Keep one body per exact fingerprint; metadata and lineage for every row."""
    def __init__(self, raw_reference=None, reserved_titles=None):
        self.raw_reference = raw_reference or {}
        self.reserved_titles = reserved_titles or {}
        self.entries = []
        self.lookup = {}
        self.lineage = []
        self.counts = Counter()
        self.parent = []
        self.edges = []

    def add(self, collection, source_id, title, text, label, published_at=None, legacy_split=""):
        self.counts[f"{collection}_raw_rows"] += 1
        title = "" if title is None or pd.isna(title) else str(title)
        text = "" if text is None or pd.isna(text) else str(text)
        row = {"collection": collection, "source_record_id": str(source_id), "label": int(label),
               "group_index": None, "initial_status": "accepted"}
        normalized = normalized_body(text)
        word_count = len(normalized.split())
        if len(normalized) < 100 or word_count < MIN_BODY_WORDS:
            row["initial_status"] = "missing_or_short_body"
            self.counts[f"{collection}_missing_or_short_body"] += 1
            self.lineage.append(row)
            return
        if len(text) > 500000:
            row["initial_status"] = "oversized_body"
            self.counts[f"{collection}_oversized_body"] += 1
            self.lineage.append(row)
            return
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        raw_bits = self.raw_reference.get(digest, 0)
        if collection == "WELFake" and raw_bits:
            self.counts["welfake_rows_matching_raw_ISOT"] += 1
        if digest in self.lookup:
            index = self.lookup[digest]
            entry = self.entries[index]
            self.counts[f"{collection}_exact_duplicate_rows"] += 1
            entry["label_bits"] |= (1 << label) | raw_bits
            entry["collections"].add(collection)
            entry["provenance_count"] += 1
        else:
            index = len(self.entries)
            self.lookup[digest] = index
            entry = {
                "hash": digest, "title": title, "text": text, "label_bits": (1 << label) | raw_bits,
                "word_count": word_count, "collections": {collection}, "provenance_count": 1,
                "published_at": published_at, "legacy_splits": set(), "reserved": set(),
                "raw_ISOT_overlap": bool(raw_bits), "has_processed_ISOT": collection == "ISOT",
            }
            self.entries.append(entry)
            self.parent.append(index)
        if collection == "ISOT":
            entry["has_processed_ISOT"] = True
        if legacy_split:
            entry["legacy_splits"].add(legacy_split)
            if legacy_split in ("validation", "test"):
                entry["reserved"].add("legacy_"+legacy_split)
        entry["reserved"].update(self.reserved_titles.get(headline_key(title), set()))
        row["group_index"] = index
        self.lineage.append(row)

    def find(self, index):
        root = index
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[index] != index:
            previous = self.parent[index]
            self.parent[index] = root
            index = previous
        return root

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def screen_near_duplicates(self):
        buckets = defaultdict(list)
        anchors = defaultdict(list)
        hashes = []
        candidate_pairs = 0
        capped_buckets = 0
        @lru_cache(maxsize=128)
        def grams(index):
            return word_shingles(self.entries[index]["text"])
        for index, entry in enumerate(self.entries):
            signature = simhash64(entry["text"])
            candidates = set()
            anchor_candidates = set()
            words = normalized_body(entry["text"]).split()
            anchor_keys = []
            if len(words) >= 40:
                anchor_keys = [
                    ("prefix", hashlib.sha256(" ".join(words[:16]).encode()).digest()[:12]),
                    ("suffix", hashlib.sha256(" ".join(words[-16:]).encode()).digest()[:12]),
                ]
            for key in anchor_keys:
                bucket = anchors.get(key, [])
                if len(bucket) <= 300:
                    anchor_candidates.update(bucket)
                else:
                    capped_buckets += 1
            candidates.update(anchor_candidates)
            for band in range(4):
                bucket = buckets.get((band, (signature >> (band*16)) & 0xffff), [])
                if len(bucket) <= 300:
                    candidates.update(bucket)
                else:
                    capped_buckets += 1
            for candidate in sorted(candidates):
                if self.find(candidate) == self.find(index):
                    continue
                distance = (signature ^ hashes[candidate]).bit_count()
                if distance > 3 and candidate not in anchor_candidates:
                    continue
                other = self.entries[candidate]
                ratio = min(entry["word_count"], other["word_count"])/max(entry["word_count"], other["word_count"])
                if ratio < .85:
                    continue
                first, second = grams(index), grams(candidate)
                candidate_pairs += 1
                denominator = len(first | second)
                similarity = len(first & second)/denominator if denominator else 0.
                if similarity >= .85:
                    self.union(index, candidate)
                    self.edges.append({"first_hash": other["hash"], "second_hash": entry["hash"],
                                       "simhash_distance": distance, "word_5_shingle_jaccard": similarity,
                                       "candidate_method": "body_anchor" if candidate in anchor_candidates else "simhash"})
            hashes.append(signature)
            for band in range(4):
                buckets[(band, (signature >> (band*16)) & 0xffff)].append(index)
            for key in anchor_keys:
                anchors[key].append(index)
            if (index+1) % 5000 == 0:
                print(f"Near-duplicate screen: {index+1}/{len(self.entries)} unique bodies", flush=True)
        self.counts["near_duplicate_edges"] = len(self.edges)
        self.counts["verified_candidate_pairs"] = candidate_pairs
        self.counts["capped_bucket_queries"] = capped_buckets
        grams.cache_clear()

    def build_rows(self):
        clusters = defaultdict(list)
        for index in range(len(self.entries)):
            clusters[self.find(index)].append(index)
        status_by_index = {}
        result = []
        quarantine = []
        for root, indices in clusters.items():
            entries = [self.entries[i] for i in indices]
            labels = 0
            collections, reserved, splits = set(), set(), set()
            for entry in entries:
                labels |= entry["label_bits"]
                collections.update(entry["collections"])
                reserved.update(entry["reserved"])
                splits.update(entry["legacy_splits"])
            identifier = "corpus-v3-" + min(entry["hash"] for entry in entries)[:24]
            has_base = any(entry["has_processed_ISOT"] for entry in entries)
            raw_overlap = any(entry["raw_ISOT_overlap"] for entry in entries)
            reason = None
            if labels == 3:
                reason = "conflicting_labels"
            elif raw_overlap and not has_base:
                reason = "raw_ISOT_overlap_not_in_processed_base"
            if reason:
                quarantine.append({"id": identifier, "reason": reason, "source_rows": sum(entry["provenance_count"] for entry in entries), "body_variants": len(indices)})
                for i in indices:
                    status_by_index[i] = (identifier, reason)
                continue
            if "external_seen" in reserved:
                eligibility = "reserved_external_seen"
            elif "legacy_test" in reserved:
                eligibility = "reserved_legacy_test"
            elif "legacy_validation" in reserved:
                eligibility = "reserved_legacy_validation"
            elif has_base:
                eligibility = "legacy_train"
            else:
                eligibility = "new_candidate_unassigned"
            chosen = min(indices, key=lambda i: (not self.entries[i]["has_processed_ISOT"], self.entries[i]["hash"]))
            representative = self.entries[chosen]
            dates = [pd.Timestamp(entry["published_at"]) for entry in entries if entry["published_at"] is not None and not pd.isna(entry["published_at"])]
            # New WELFake rows have no supplied publication date; never fabricate one.
            published_at = min(dates) if dates else None
            result.append({
                "id": identifier, "group_id": min(entry["hash"] for entry in entries),
                "title": representative["title"], "text": representative["text"],
                "label": 0 if labels == 1 else 1, "published_at": published_at,
                "collections": sorted(collections), "original_publisher": None,
                "provenance_rows": sum(entry["provenance_count"] for entry in entries),
                "body_variants": len(indices), "eligibility": eligibility,
                "legacy_splits": sorted(splits), "has_processed_ISOT": has_base,
                "raw_ISOT_overlap": raw_overlap, "split": "unassigned",
            })
            for i in indices:
                status_by_index[i] = (identifier, eligibility)
        for row in self.lineage:
            index = row["group_index"]
            row["merged_id"], row["final_status"] = status_by_index[index] if index is not None else (None, row["initial_status"])
        return result, quarantine


def import_welfake(csv_path=None, output_dir=OUTPUT_DIR, report_dir=REPORT_DIR):
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_path) if csv_path else download_welfake()
    active_model = PROJECT_DIR / "artifacts" / "model.joblib"
    original_corpus = PROJECT_DIR / "data" / "processed" / "articles.parquet"
    protected_hashes = {
        "active_model": file_sha256(active_model), "original_ISOT_corpus": file_sha256(original_corpus),
        "active_version_pointer": file_sha256(PROJECT_DIR / "artifacts" / "active_version.json"),
    }
    base = load_articles()
    training_labels = {}
    for row in base[base.split == "train"].itertuples():
        digest = body_hash(row.text)
        training_labels[digest] = training_labels.get(digest, 0) | (1 << int(row.label))
    print("Auditing WELFake label orientation against identical TRAIN articles...", flush=True)
    mapping_path = report_dir / "label_mapping_audit.json"
    csv_hash = file_sha256(csv_path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
    if not (
        mapping.get("raw_csv_sha256") == csv_hash
        and mapping.get("ISOT_corpus_sha256") == protected_hashes["original_ISOT_corpus"]
        and mapping.get("audit_method") == "train-exact-copy-label-orientation-v1"
    ):
        mapping = audit_label_mapping(csv_path, training_labels)
        mapping.update({"raw_csv_sha256": csv_hash, "ISOT_corpus_sha256": protected_hashes["original_ISOT_corpus"],
                        "audit_method": "train-exact-copy-label-orientation-v1"})
    write_json(report_dir / "label_mapping_audit.json", mapping)
    print(f"Label convention: {mapping['effective_convention']}; TRAIN evidence={mapping['evidence']['unique_training_copies']}", flush=True)
    print("Fingerprinting original ISOT files...", flush=True)
    reference = raw_isot_reference()
    reserved = defaultdict(set)
    for row in base[base.split.isin(["validation", "test"])].itertuples():
        key = headline_key(row.title)
        if key:
            reserved[key].add("legacy_"+row.split)
    for external_file in (PROJECT_DIR / "data" / "raw" / "fakenewsnet").glob("*.csv"):
        external = pd.read_csv(external_file, usecols=["title"], keep_default_na=False)
        for title in external.title:
            key = headline_key(title)
            if key:
                reserved[key].add("external_seen")
    builder = CorpusBuilder(reference, reserved)
    for row in base.itertuples():
        builder.add("ISOT", row.id, row.title, row.text, int(row.label), row.published_at, row.split)
    parsed_rows = 0
    source_label_counts = Counter()
    for chunk in pd.read_csv(csv_path, keep_default_na=False, dtype=str, chunksize=2000):
        if not {"title", "text", "label"}.issubset(chunk.columns):
            raise ValueError("WELFake CSV missing title/text/label columns.")
        for _, row in chunk.iterrows():
            ordinal = parsed_rows
            parsed_rows += 1
            raw_label = row["label"]
            try:
                label = normalize_label(raw_label, mapping["effective_convention"])
            except ValueError:
                builder.counts["WELFake_raw_rows"] += 1
                builder.counts["WELFake_invalid_labels"] += 1
                builder.lineage.append({"collection": "WELFake", "source_record_id": str(ordinal), "label": None, "group_index": None, "initial_status": "invalid_label"})
                continue
            source_label_counts[str(raw_label)] += 1
            builder.add("WELFake", ordinal, row["title"], row["text"], label)
        if parsed_rows % 10000 == 0:
            print(f"Imported {parsed_rows} WELFake records; exact unique bodies={len(builder.entries)}", flush=True)
    print("Checking verified near-duplicate body clusters...", flush=True)
    builder.screen_near_duplicates()
    rows, quarantine = builder.build_rows()
    # Serialize in batches to keep working memory bounded on a 6-GB laptop.
    schema = pa.schema([
        ("id", pa.string()), ("group_id", pa.string()), ("title", pa.string()), ("text", pa.string()),
        ("label", pa.int8()), ("published_at", pa.timestamp("ns")),
        ("collections", pa.list_(pa.string())), ("original_publisher", pa.string()),
        ("provenance_rows", pa.int32()), ("body_variants", pa.int32()),
        ("eligibility", pa.string()), ("legacy_splits", pa.list_(pa.string())),
        ("has_processed_ISOT", pa.bool_()), ("raw_ISOT_overlap", pa.bool_()), ("split", pa.string()),
    ])
    corpus_path = output_dir / "articles.parquet"
    temporary = output_dir / "articles.tmp.parquet"
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for start in range(0, len(rows), 1000):
            writer.write_table(pa.Table.from_pylist(rows[start:start+1000], schema=schema))
    temporary.replace(corpus_path)
    lineage = pd.DataFrame(builder.lineage)
    lineage.drop(columns=["group_index"]).to_parquet(output_dir / "provenance.parquet", index=False)
    pd.DataFrame(quarantine, columns=["id", "reason", "source_rows", "body_variants"]).to_csv(report_dir / "quarantine.csv", index=False)
    pd.DataFrame(builder.edges, columns=["first_hash", "second_hash", "simhash_distance", "word_5_shingle_jaccard", "candidate_method"]).to_csv(report_dir / "near_duplicate_edges.csv", index=False)
    eligibility = Counter(row["eligibility"] for row in rows)
    class_counts = Counter(str(row["label"]) for row in rows)
    fresh = [row for row in rows if not row["has_processed_ISOT"] and not row["raw_ISOT_overlap"]]
    source_mix = Counter("+".join(row["collections"]) for row in rows)
    report = {
        "version": "corpus-v3-candidate", "method_version": METHOD_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ISOT_processed_input_rows": len(base), "WELFake_parsed_input_rows": parsed_rows,
        "WELFake_raw_label_counts": dict(source_label_counts), "label_mapping": mapping,
        "processing_counts": dict(builder.counts), "exact_unique_body_groups": len(builder.entries),
        "retained_unique_article_clusters": len(rows),
        "new_unique_clusters_vs_all_raw_ISOT": len(fresh),
        "new_candidates_unassigned": eligibility["new_candidate_unassigned"],
        "eligibility_counts": dict(eligibility), "retained_label_counts": dict(class_counts),
        "collection_combinations": dict(source_mix), "quarantined_clusters": len(quarantine),
        "quarantine_reasons": dict(Counter(row["reason"] for row in quarantine)),
        "minimum_body_words": MIN_BODY_WORDS,
        "deduplication": {
            "exact": "SHA-256 of NFKC/lowercased publisher-cleaned body word sequence; includes negation.",
            "near_candidate": "64-bit SimHash (4x16-bit bands, Hamming <=3) OR identical first/last 16-word body anchors; length ratio >=0.85.",
            "near_confirmation": "Word 5-shingle Jaccard >=0.85; connected components.",
            "limit": "Approximate candidate search, max 300 entries per queried bucket; not exhaustive semantic/paraphrase deduplication.",
        },
        "partition_policy": "Candidate data only, split=unassigned. Frozen ISOT validation/test and already-seen external headline matches are reserved, including near-body clusters.",
        "missing_dates": "WELFake supplies no publication dates; new records keep null, never fabricated dates.",
        "publisher_provenance": "Collection names are recorded. WELFake does not identify the original publisher per row; it is not counted as a new independent publisher.",
        "rights": "Local research import. Public redistribution rights not established.",
        "frozen_model_retrained": False,
        "protected_hashes_before": protected_hashes,
    }
    now = {
        "active_model": file_sha256(active_model), "original_ISOT_corpus": file_sha256(original_corpus),
        "active_version_pointer": file_sha256(PROJECT_DIR / "artifacts" / "active_version.json"),
    }
    if now != protected_hashes:
        raise RuntimeError("Import modified a protected frozen artifact.")
    report["protected_artifacts_unchanged"] = True
    manifest = {
        "version": report["version"], "method_version": METHOD_VERSION,
        "source_record": "https://zenodo.org/records/4561253",
        "raw_welfake_sha256": file_sha256(csv_path), "raw_welfake_md5": WELFAKE_MD5,
        "input_ISOT_sha256": protected_hashes["original_ISOT_corpus"],
        "outputs": {name: file_sha256(output_dir / name) for name in ("articles.parquet", "provenance.parquet")},
        "implementation": {
            "news15/corpus_expansion.py": file_sha256(Path(__file__)),
            "scripts/import_welfake.py": file_sha256(PROJECT_DIR / "scripts" / "import_welfake.py"),
        },
        "label_convention": mapping["effective_convention"], "training_ready": False,
        "reason": "Requires a new registered split/protocol; publisher independence and missing dates remain unresolved.",
    }
    write_json(report_dir / "import_report.json", report)
    write_json(output_dir / "manifest.json", manifest)
    lines = [
        "# Импорт WELFake и проверка пересечений", "",
        f"Исходный ISOT: {len(base):,} записей. WELFake: {parsed_rows:,} CSV-записей.",
        f"После фильтров, конфликтов и дедупликации: {len(rows):,} уникальных кластеров.",
        f"Новых относительно всех raw ISOT: {len(fresh):,}; новых кандидатов без назначенного разбиения: {eligibility['new_candidate_unassigned']:,}.",
        "", "## Метки", "",
        f"Эффективная схема WELFake: {mapping['effective_convention']}.",
        f"Документированное обратное кодирование согласуется с {mapping['documented_inverse_agreement_rate']:.2%} точных TRAIN-копий; прямое — с {mapping['identity_agreement_rate']:.2%}.",
        "Схема проверена по идентичным статьям существующего train, без прогнозов и без использования test для подбора меток.",
        "Расхождение документации и файла явно сохранено в label_mapping_audit.json.",
        "", "## Защита проверки", "",
        "Прежние validation/test и совпадения с уже просмотренными внешними заголовками зарезервированы.",
        "Кластеры с конфликтующими метками помещены в карантин, а не исправлены автоматически.",
        "У WELFake нет дат публикации и издателя на уровне записи. Даты не выдуманы; коллекция не приравнена к независимому СМИ.",
        "", "## Файлы", "",
        "data/processed/corpus_v3/articles.parquet — отдельный объединённый кандидатный корпус.",
        "data/processed/corpus_v3/provenance.parquet — происхождение и решение для каждой входной строки.",
        "data/processed/corpus_v3/manifest.json — SHA-256 исходников и выходных данных.",
        "reports/corpus_expansion/import_report.json — численные результаты и ограничения.",
        "", "Зафиксированная модель, прежний корпус и указатель версии не изменены. Нового обучения и новой оценки качества не проводилось.",
    ]
    (report_dir / "IMPORT_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("WELFake_parsed_input_rows", "retained_unique_article_clusters", "new_unique_clusters_vs_all_raw_ISOT", "new_candidates_unassigned", "quarantined_clusters", "protected_artifacts_unchanged")}, indent=2), flush=True)
    return report

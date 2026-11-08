"""Download the authors' ISOT archive and prepare a reproducible temporal split."""

import hashlib
import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.base import BaseEstimator, TransformerMixin

from .paths import RAW_DIR, PROCESSED_DIR, ensure_directories

ISOT_URL = (
    "https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/"
    "sites/7295/2023/03/News-_dataset.zip"
)
ISOT_DESCRIPTION = (
    "https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/"
    "fake-news-detection-datasets/"
)
PREPROCESSING_VERSION = "isot-temporal-v1"
ARTICLE_PATH = PROCESSED_DIR / "articles.parquet"
SUMMARY_PATH = PROCESSED_DIR / "dataset_summary.json"

_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
_REUTERS_HEADER = re.compile(r"^.{0,100}?\(\s*reuters\s*\)\s*[-–—:]*\s*", re.I)
_REUTERS = re.compile(r"\b(?:reuters(?:\.com)?|thomson reuters)\b", re.I)
_HTML = re.compile(r"<[^>]+>")


def clean_text(text):
    """Remove explicit publisher/URL shortcuts, preserve negation and word order."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _REUTERS_HEADER.sub("", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _REUTERS.sub(" ", text)
    text = _HTML.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


class TextCleaner(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([clean_text(text) for text in X], dtype=object)


def combine_text(title, text):
    title = "" if title is None or pd.isna(title) else str(title).strip()
    text = "" if text is None or pd.isna(text) else str(text).strip()
    return f"{title}. {text}".strip(". ")


def article_texts(frame):
    return [
        combine_text(title, text)
        for title, text in zip(frame["title"], frame["text"])
    ]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_isot(raw_dir=RAW_DIR, force=False):
    """Fetch original data, safely extract only the two expected CSV files."""
    ensure_directories()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "News_dataset.zip"
    if force or not archive.exists() or not is_zipfile(archive):
        partial = raw_dir / "News_dataset.zip.part"
        try:
            with requests.get(ISOT_URL, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if not is_zipfile(partial):
                raise ValueError("The dataset server did not return a valid ZIP archive.")
            partial.replace(archive)
        finally:
            if partial.exists():
                partial.unlink()
    with ZipFile(archive) as source:
        for expected in ("True.csv", "Fake.csv"):
            matches = [
                info for info in source.infolist()
                if Path(info.filename).name.lower() == expected.lower()
                and not info.is_dir()
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {expected} in the ISOT archive.")
            info = matches[0]
            if info.file_size > 400 * 1024 * 1024:
                raise ValueError("Unexpectedly large CSV in dataset archive.")
            target = raw_dir / expected
            if force or not target.exists() or target.stat().st_size != info.file_size:
                with source.open(info) as input_stream, target.open("wb") as output:
                    shutil.copyfileobj(input_stream, output)
    metadata = {
        "dataset": "ISOT Fake News Dataset",
        "source_url": ISOT_URL,
        "description_url": ISOT_DESCRIPTION,
        "archive_sha256": file_sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            name: {"sha256": file_sha256(raw_dir / name), "bytes": (raw_dir / name).stat().st_size}
            for name in ("True.csv", "Fake.csv")
        },
    }
    (raw_dir / "source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return archive


def split_articles(frame, validation_fraction=0.15, test_fraction=0.15):
    """Split on whole calendar dates; no future article enters training."""
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")
    result = frame.sort_values(["published_at", "group_id"]).reset_index(drop=True).copy()
    if len(result) < 20:
        raise ValueError("At least 20 dated articles are required for a temporal split.")
    dates = result["published_at"].dt.normalize()
    if dates.nunique() < 3:
        raise ValueError("At least three different publication dates are required.")
    val_index = int(len(result) * (1 - validation_fraction - test_fraction))
    test_index = int(len(result) * (1 - test_fraction))
    validation_start = dates.iloc[val_index]
    test_start = dates.iloc[test_index]
    if validation_start >= test_start:
        unique_dates = dates.drop_duplicates().sort_values().tolist()
        validation_start = unique_dates[max(1, int(len(unique_dates) * 0.70))]
        test_start = unique_dates[min(len(unique_dates) - 1, max(2, int(len(unique_dates) * 0.85)))]
    result["split"] = np.where(
        dates < validation_start, "train",
        np.where(dates < test_start, "validation", "test"),
    )
    for name in ("train", "validation", "test"):
        part = result.loc[result["split"] == name]
        if part.empty or part["label"].nunique() != 2:
            raise ValueError(f"Temporal {name} split must contain both labels.")
    return result


def prepare_isot(raw_dir=RAW_DIR, output_path=None):
    ensure_directories()
    raw_dir = Path(raw_dir)
    if not all((raw_dir / name).exists() for name in ("True.csv", "Fake.csv")):
        download_isot(raw_dir)
    parts = []
    for filename, label in (("True.csv", 0), ("Fake.csv", 1)):
        part = pd.read_csv(raw_dir / filename, keep_default_na=False)
        expected = {"title", "text", "subject", "date"}
        if not expected.issubset(part.columns):
            raise ValueError(f"{filename} is missing required columns: {expected - set(part.columns)}")
        part = part[list(sorted(expected))].copy()
        part["label"] = label
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    raw_rows = len(frame)
    frame["published_at"] = pd.to_datetime(
        frame["date"].astype(str).str.strip(), format="mixed", errors="coerce", utc=True
    ).dt.tz_convert(None)
    missing_dates = int(frame["published_at"].isna().sum())
    frame = frame.dropna(subset=["published_at"]).copy()
    cleaned_body = frame["text"].map(clean_text)
    fallback = frame["title"].map(clean_text)
    canonical = cleaned_body.where(cleaned_body.str.len() >= 40, fallback)
    usable = canonical.str.len() >= 20
    removed_empty = int((~usable).sum())
    frame = frame.loc[usable].copy()
    canonical = canonical.loc[usable]
    frame["group_id"] = canonical.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    )
    contradictory = frame.groupby("group_id")["label"].nunique()
    ambiguous_ids = contradictory[contradictory > 1].index
    ambiguous_rows = int(frame["group_id"].isin(ambiguous_ids).sum())
    frame = frame.loc[~frame["group_id"].isin(ambiguous_ids)].copy()
    before_dedup = len(frame)
    frame = frame.sort_values("published_at").drop_duplicates("group_id", keep="first")
    removed_duplicates = before_dedup - len(frame)
    frame["id"] = "isot-" + frame["group_id"]
    frame = split_articles(frame)
    columns = ["id", "title", "text", "label", "subject", "date", "published_at", "split", "group_id"]
    frame = frame[columns].reset_index(drop=True)
    output_path = Path(output_path or ARTICLE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    source_path = raw_dir / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    summary = {
        "dataset": "ISOT",
        "preprocessing_version": PREPROCESSING_VERSION,
        "raw_rows": raw_rows,
        "clean_rows": len(frame),
        "removed_duplicates": removed_duplicates,
        "ambiguous_groups": len(ambiguous_ids),
        "ambiguous_rows": ambiguous_rows,
        "missing_dates": missing_dates,
        "removed_empty": removed_empty,
        "split_counts": {k: int(v) for k, v in frame["split"].value_counts().items()},
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        "date_range": {
            "start": frame["published_at"].min().isoformat(),
            "end": frame["published_at"].max().isoformat(),
        },
        "split_details": {
            name: {
                "rows": len(part),
                "real": int((part["label"] == 0).sum()),
                "fake": int((part["label"] == 1).sum()),
                "start": part["published_at"].min().isoformat(),
                "end": part["published_at"].max().isoformat(),
            }
            for name, part in frame.groupby("split")
        },
        "language": "English",
        "source": source,
        "limitations": [
            "Source-confounded labels: real articles from Reuters, fake articles from other outlets.",
            "The dataset does not provide a verified publisher identifier for every article.",
            "Chronological evaluation is not a source-disjoint or multilingual evaluation.",
            "Removing explicit source markers does not remove all stylistic shortcuts.",
        ],
    }
    summary_path = SUMMARY_PATH if output_path == ARTICLE_PATH else output_path.parent / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame


def load_articles(path=ARTICLE_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Dataset is not prepared. Run: python scripts/download_data.py")
    return pd.read_parquet(path)


def dataset_summary():
    if not SUMMARY_PATH.exists():
        return {"status": "missing", "raw_rows": 0, "clean_rows": 0}
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

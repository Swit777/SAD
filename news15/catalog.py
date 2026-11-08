"""Small catalog summaries and projected Parquet reads for the Streamlit UI."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .paths import PROJECT_DIR


SCOPE_LABELS = {
    "collection_annotation": "Метка исходной коллекции",
    "publisher_proxy": "Категория сайта",
    "reporting_quality": "Качество освещения",
    "claim_factcheck": "Оценка утверждения",
    "unlabelled_claim": "Метка не опубликована",
}
SOURCE_LABELS = {"ISOT+WELFake":"ISOT и WELFake", "WELFake":"WELFake", "FakeNewsCorpus":"FakeNewsCorpus", "FakeHealth":"FakeHealth", "LIAR":"LIAR", "MultiFC":"MultiFC"}


def catalog_summary(project_dir=PROJECT_DIR):
    root = Path(project_dir)
    report_path = root/"reports/multisource_expansion/import_report.json"
    validation_path = root/"reports/multisource_expansion/validation.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    articles_path = root/"data/processed/corpus_v4/articles.parquet"
    claims_path = root/"data/processed/corpus_v4/claims.parquet"
    article_rows = pq.ParquetFile(articles_path).metadata.num_rows if articles_path.exists() else 0
    claim_rows = pq.ParquetFile(claims_path).metadata.num_rows if claims_path.exists() else 0
    counts_agree = article_rows == report["article_rows"] and claim_rows == report["claims"]["records"]
    counts = report["new_source_counts"]
    sources = [
        {"Источник":"ISOT и WELFake", "Единица":"Текст", "Записей":report["base_v3_articles"], "Тип разметки":"Исходные метки коллекций", "Применение":"Базовый корпус v3"},
        {"Источник":"FakeNewsCorpus", "Единица":"Текст", "Записей":counts["FakeNewsCorpus_retained"], "Тип разметки":"Категория сайта", "Применение":"Отдельные эксперименты со слабой разметкой"},
        {"Источник":"FakeHealth", "Единица":"Текст", "Записей":counts["FakeHealth_retained"], "Тип разметки":"Качество освещения", "Применение":"Анализ медицинских публикаций"},
        {"Источник":"LIAR", "Единица":"Утверждение", "Записей":report["claims"]["collection_records"]["LIAR"], "Тип разметки":"Шесть оценок PolitiFact", "Применение":"Отдельная задача проверки утверждений"},
        {"Источник":"MultiFC", "Единица":"Утверждение", "Записей":report["claims"]["collection_records"]["MultiFC"], "Тип разметки":"Шкалы фактчекеров и тест без меток", "Применение":"Отдельная задача проверки утверждений"},
    ]
    return {"article_rows":article_rows,"claim_rows":claim_rows,"new_article_rows":report["new_article_rows"],
            "claim_groups":report["claims"]["exact_unique_claim_groups"],"domains":report["FakeNewsCorpus_domains"],
            "sources":sources,"version":report["version"],"validated_at":validation.get("validated_at_utc"),
            "validation_passed":bool(validation.get("passed") and counts_agree),"counts_agree":counts_agree,
            "checks_passed":sum(bool(v) for v in validation.get("checks",{}).values()),
            "training_ready":report["training_ready"],"previous_rows":report["base_v3_articles"]}


def preview_records(kind="articles", source="Все", query="", limit=30, project_dir=PROJECT_DIR):
    if kind not in {"articles","claims"}:
        raise ValueError("Unknown catalog table.")
    if not 1 <= limit <= 100:
        raise ValueError("Preview limit must be between 1 and 100.")
    path = Path(project_dir)/"data/processed/corpus_v4"/(kind+".parquet")
    file = pq.ParquetFile(path)
    wanted = ["id","collection","title" if kind == "articles" else "text","original_label","label_scope","eligibility"]
    wanted += ["publisher","published_at"] if kind == "articles" else ["original_split"]
    rows = []
    needle = query.strip().casefold()
    collection_index = file.schema_arrow.get_field_index("collection")
    for group in range(file.num_row_groups):
        if source != "Все":
            stats = file.metadata.row_group(group).column(collection_index).statistics
            if stats and stats.has_min_max and (source < stats.min or source > stats.max):
                continue
        # Body texts of articles are never loaded for listing or searching.
        for row in file.read_row_group(group,columns=wanted).to_pylist():
            if source != "Все" and row["collection"] != source:
                continue
            text = row.get("title") or row.get("text") or ""
            if needle and needle not in text.casefold() and needle not in row["id"].casefold():
                continue
            row["_row_group"] = group
            rows.append(row)
            if len(rows) == limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def catalog_record(kind, identifier, row_group, project_dir=PROJECT_DIR):
    if kind not in {"articles","claims"}:
        raise ValueError("Unknown catalog table.")
    path = Path(project_dir)/"data/processed/corpus_v4"/(kind+".parquet")
    file = pq.ParquetFile(path)
    if not 0 <= int(row_group) < file.num_row_groups:
        raise ValueError("Unknown row group.")
    for row in file.read_row_group(int(row_group)).to_pylist():
        if row["id"] == identifier:
            return row
    raise LookupError("Record was not found in the selected row group.")

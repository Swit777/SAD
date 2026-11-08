import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from news15 import catalog


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = self.root/"data/processed/corpus_v4"
        self.directory.mkdir(parents=True)
        rows = [
            {"id":"base","collection":"ISOT+WELFake","title":"Original headline","text":"A full original body with a hidden NEEDLE", "original_label":"0","label_scope":"collection_annotation","eligibility":"legacy_train","publisher":None,"published_at":None,"label":0},
            {"id":"new","collection":"FakeNewsCorpus","title":"Climate and health","text":"A full newly collected article", "original_label":"fake","label_scope":"publisher_proxy","eligibility":"candidate_unassigned","publisher":"example.org","published_at":None,"label":None},
        ]
        pq.write_table(pa.Table.from_pylist(rows),self.directory/"articles.parquet",row_group_size=1)
        pq.write_table(pa.Table.from_pylist([{"id":"claim","collection":"MultiFC","text":"An original claim","original_label":None,"label_scope":"unlabelled_claim","eligibility":"reserved_original_test","original_split":"test","label":None}]),self.directory/"claims.parquet")

    def tearDown(self):
        self.temporary.cleanup()

    def test_search_uses_titles_and_literal_case_insensitive_queries(self):
        result = catalog.preview_records("articles","FakeNewsCorpus","CLIMATE",project_dir=self.root)
        self.assertEqual(result.id.tolist(),["new"])
        self.assertTrue(catalog.preview_records("articles",query="NEEDLE",project_dir=self.root).empty)
        self.assertTrue(catalog.preview_records("articles",query=".*",project_dir=self.root).empty)

    def test_listing_projects_metadata_without_article_bodies(self):
        file = pq.ParquetFile(self.directory/"articles.parquet")
        reader = Mock(wraps=file.read_row_group)
        proxy = SimpleNamespace(metadata=file.metadata,schema_arrow=file.schema_arrow,num_row_groups=file.num_row_groups,read_row_group=reader)
        with patch.object(catalog.pq,"ParquetFile",return_value=proxy):
            result = catalog.preview_records(project_dir=self.root)
        self.assertEqual(len(result),2)
        self.assertTrue(all("text" not in call.kwargs["columns"] for call in reader.call_args_list))
        file.close()

    def test_record_lookup_preserves_weak_label_and_unlabelled_test(self):
        article = catalog.catalog_record("articles","new",1,self.root)
        self.assertIsNone(article["label"])
        self.assertEqual(article["original_label"],"fake")
        claim = catalog.catalog_record("claims","claim",0,self.root)
        self.assertIsNone(claim["original_label"])
        with self.assertRaises(ValueError):
            catalog.catalog_record("articles","new",-1,self.root)

    def test_summary_does_not_show_stale_validation_as_passed(self):
        reports = self.root/"reports/multisource_expansion"
        reports.mkdir(parents=True)
        report = {"article_rows":3,"new_article_rows":1,"base_v3_articles":1,"FakeNewsCorpus_domains":1,
                  "new_source_counts":{"FakeNewsCorpus_retained":1,"FakeHealth_retained":0},
                  "claims":{"records":1,"exact_unique_claim_groups":1,"collection_records":{"LIAR":0,"MultiFC":1}},
                  "version":"candidate","training_ready":False}
        (reports/"import_report.json").write_text(json.dumps(report),encoding="utf-8")
        (reports/"validation.json").write_text(json.dumps({"passed":True,"checks":{"counts":True}}),encoding="utf-8")
        summary = catalog.catalog_summary(self.root)
        self.assertEqual(summary["article_rows"],2)
        self.assertFalse(summary["counts_agree"])
        self.assertFalse(summary["validation_passed"])


if __name__ == "__main__":
    unittest.main()

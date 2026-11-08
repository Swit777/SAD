"""Data-import checks independent of full corpora and frozen model artifacts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.corpus_expansion import (
    CorpusBuilder, audit_label_mapping, body_hash, headline_key,
    normalize_label, normalized_body,
)

BODY = (
    "The city council discussed the annual education budget in a public meeting. "
    "Officials reviewed documents describing school maintenance and teacher support. "
    "Residents asked several questions about costs and the planned timetable. "
    "A detailed report will be available after the next committee session."
)


class LabelEncodingTests(unittest.TestCase):
    def test_both_explicit_conventions_and_invalid_labels(self):
        self.assertEqual(normalize_label("0"), 1)
        self.assertEqual(normalize_label("1"), 0)
        self.assertEqual(normalize_label("0", "verified_identity"), 0)
        self.assertEqual(normalize_label("1", "verified_identity"), 1)
        for bad in ("", "fake", "true", "0.0", "2"):
            with self.assertRaises(ValueError):
                normalize_label(bad)

    def test_documentation_discrepancy_is_detected_from_unique_train_copies(self):
        other = BODY.replace("education", "transport").replace("school", "road")
        training = {body_hash(BODY): 1, body_hash(other): 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "welfake.csv"
            pd.DataFrame({"text": [BODY, other, BODY], "label": ["0", "1", "0"]}).to_csv(path, index=False)
            report = audit_label_mapping(path, training, min_evidence=2)
            self.assertEqual(report["effective_convention"], "verified_identity")
            self.assertTrue(report["documentation_discrepancy"])
            self.assertEqual(report["evidence"]["unique_training_copies"], 2)
            self.assertEqual(report["identity_agreement_rate"], 1)

    def test_repeated_copies_cannot_artificially_satisfy_encoding_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "welfake.csv"
            pd.DataFrame({"text": [BODY]*8, "label": ["0"]*8}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                audit_label_mapping(path, {body_hash(BODY): 1}, min_evidence=2)

    def test_normalization_keeps_negation_and_numbers(self):
        self.assertNotEqual(body_hash(BODY), body_hash(BODY.replace("will be", "will not be")))
        self.assertNotEqual(body_hash(BODY+" Cost 100"), body_hash(BODY+" Cost 1000"))
        self.assertEqual(body_hash("WASHINGTON (Reuters) - "+BODY), body_hash(BODY.upper()))


class CorpusIntegrityTests(unittest.TestCase):
    def test_exact_copies_keep_provenance_and_reserve_previous_test(self):
        builder = CorpusBuilder()
        builder.add("ISOT", "old-test", "Budget report", BODY, 0, pd.Timestamp("2017-01-01"), "test")
        builder.add("WELFake", "copy", "Retitled", BODY.upper(), 0)
        rows, quarantine = builder.build_rows()
        self.assertEqual(len(rows), 1)
        self.assertFalse(quarantine)
        self.assertEqual(rows[0]["collections"], ["ISOT", "WELFake"])
        self.assertEqual(rows[0]["provenance_rows"], 2)
        self.assertEqual(rows[0]["eligibility"], "reserved_legacy_test")
        self.assertEqual(rows[0]["split"], "unassigned")
        self.assertEqual({row["merged_id"] for row in builder.lineage}, {rows[0]["id"]})

    def test_conflicting_copies_are_quarantined_not_majority_voted(self):
        builder = CorpusBuilder()
        builder.add("ISOT", "base", "Budget", BODY, 0, legacy_split="train")
        for index in range(4):
            builder.add("WELFake", index, "Budget", BODY, 1)
        rows, quarantine = builder.build_rows()
        self.assertFalse(rows)
        self.assertEqual(quarantine[0]["reason"], "conflicting_labels")
        self.assertEqual(quarantine[0]["source_rows"], 5)

    def test_raw_isot_overlap_is_not_counted_as_an_independent_new_article(self):
        builder = CorpusBuilder({body_hash(BODY): 1})
        builder.add("WELFake", "raw-copy", "Budget", BODY, 0)
        rows, quarantine = builder.build_rows()
        self.assertFalse(rows)
        self.assertEqual(quarantine[0]["reason"], "raw_ISOT_overlap_not_in_processed_base")

    def test_seen_external_headline_is_reserved_even_with_a_different_body(self):
        reserved = {headline_key("Already evaluated headline"): {"external_seen"}}
        builder = CorpusBuilder(reserved_titles=reserved)
        builder.add("WELFake", 1, "Already evaluated headline", BODY, 0)
        rows, _ = builder.build_rows()
        self.assertEqual(rows[0]["eligibility"], "reserved_external_seen")
        self.assertIsNone(rows[0]["published_at"])
        self.assertIsNone(rows[0]["original_publisher"])

    def test_missing_body_is_not_replaced_with_a_headline(self):
        builder = CorpusBuilder()
        builder.add("WELFake", 1, "This is only a headline", "", 0)
        rows, _ = builder.build_rows()
        self.assertFalse(rows)
        self.assertEqual(builder.lineage[0]["final_status"], "missing_or_short_body")

    def test_verified_near_body_copies_share_a_reserved_cluster(self):
        builder = CorpusBuilder()
        builder.add("ISOT", "test", "One headline", BODY, 0, legacy_split="test")
        builder.add("WELFake", "edited-copy", "A different headline", BODY+" Minor update.", 0)
        builder.screen_near_duplicates()
        rows, _ = builder.build_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["body_variants"], 2)
        self.assertEqual(rows[0]["eligibility"], "reserved_legacy_test")
        self.assertGreaterEqual(builder.edges[0]["word_5_shingle_jaccard"], .85)


if __name__ == "__main__":
    unittest.main()

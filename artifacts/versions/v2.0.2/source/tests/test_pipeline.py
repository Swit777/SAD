"""Small offline regression checks; corpus metrics are a separate experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news15 import data
from news15 import model as model_module
from news15.model import explain_article, load_bundle, predict_article, train_models
from news15.streaming import AdaptiveReviewPolicy
from scripts.evaluate_stream import evaluate_replay


def sample_articles() -> pd.DataFrame:
    rows = []
    for index in range(120):
        label = index % 2
        split = "train" if index < 84 else "validation" if index < 102 else "test"
        words = "official council report budget approved" if label == 0 else "secret conspiracy shocking miracle rumor"
        heldout = " heldoutonlytoken" if split != "train" else ""
        rows.append({
            "id": f"sample-{index}",
            "title": f"Article {index}",
            "text": f"{words}. The public discusses this detailed article with evidence number {index}.{heldout}",
            "label": label,
            "subject": "politics" if label == 0 else "news",
            "published_at": pd.Timestamp("2017-01-01") + pd.Timedelta(days=index // 2),
            "split": split,
            "group_id": f"group-{index}",
        })
    return pd.DataFrame(rows)


class DataIntegrityTests(unittest.TestCase):
    def test_cleaning_removes_shortcuts_but_preserves_negation(self):
        text = "WASHINGTON (Reuters) - <b>The council has not approved</b> it. https://example.org/story"
        cleaned = data.TextCleaner().fit_transform([text])[0]
        self.assertIn("has not approved", cleaned)
        self.assertNotIn("reuters", cleaned)
        self.assertNotIn("example.org", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertEqual(data.combine_text(None, "Body"), "Body")

    def test_preparation_prevents_duplicate_and_future_leakage(self):
        articles = sample_articles()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_parts = {}
            for label, name in ((0, "True.csv"), (1, "Fake.csv")):
                part = articles.loc[articles["label"].eq(label), ["title", "text", "subject", "published_at"]].copy()
                part["date"] = part.pop("published_at").dt.strftime("%B %d, %Y")
                csv_parts[name] = part
            # The same real story appears again after the intended split boundary.
            duplicate = csv_parts["True.csv"].iloc[[0]].copy()
            duplicate["title"] = "Retitled old story"
            duplicate["date"] = "December 30, 2017"
            csv_parts["True.csv"] = pd.concat([csv_parts["True.csv"], duplicate], ignore_index=True)
            # Conflicting labels must be removed, rather than assigned by file order.
            conflicting_body = "This deliberately conflicting article has a sufficiently long body to identify the same story."
            for name in csv_parts:
                extra = pd.DataFrame([{
                    "title": "Conflicting story", "text": conflicting_body,
                    "subject": "news", "date": "January 01, 2017",
                }])
                csv_parts[name] = pd.concat([csv_parts[name], extra], ignore_index=True)
            invalid = pd.DataFrame([{
                "title": "Undated", "text": "An undated article should never silently enter chronological testing.",
                "subject": "news", "date": "not-a-date",
            }])
            csv_parts["Fake.csv"] = pd.concat([csv_parts["Fake.csv"], invalid], ignore_index=True)
            for name, part in csv_parts.items():
                part.to_csv(root / name, index=False)
            prepared = data.prepare_isot(raw_dir=root, output_path=root / "articles.parquet")
            summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(prepared), len(articles))
            self.assertEqual(summary["removed_duplicates"], 1)
            self.assertEqual(summary["ambiguous_rows"], 2)
            self.assertEqual(summary["missing_dates"], 1)
            self.assertTrue(prepared["group_id"].is_unique)
            self.assertNotIn("Retitled old story", set(prepared["title"]))
            train = prepared.loc[prepared["split"].eq("train")]
            validation = prepared.loc[prepared["split"].eq("validation")]
            test = prepared.loc[prepared["split"].eq("test")]
            self.assertLess(train["published_at"].max(), validation["published_at"].min())
            self.assertLess(validation["published_at"].max(), test["published_at"].min())
            for left, right in ((train, validation), (train, test), (validation, test)):
                self.assertFalse(set(left["group_id"]) & set(right["group_id"]))
            for part in (train, validation, test):
                self.assertEqual(set(part["label"]), {0, 1})


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "model.joblib"
        cls.articles = sample_articles()
        with patch.object(model_module, "REPORTS_DIR", Path(cls.temp.name), create=True):
            cls.bundle = train_models(articles=cls.articles, max_features=200, output_path=cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_vectorizer_does_not_see_heldout_vocabulary(self):
        vocabulary = self.bundle["pipeline"].named_steps["tfidf"].vocabulary_
        self.assertNotIn("heldoutonlytoken", vocabulary)
        self.assertIn("official", vocabulary)
        self.assertIn("conspiracy", vocabulary)

    def test_saved_model_preserves_predictions(self):
        loaded = load_bundle(self.path)
        title, text = "Council report", "The official council report states that the budget was approved."
        first = predict_article(self.bundle, title, text)
        restored = predict_article(loaded, title, text)
        self.assertAlmostEqual(first["probability_fake"], restored["probability_fake"], places=12)
        self.assertTrue(0 <= first["probability_fake"] <= 1)

    def test_training_rejects_group_and_temporal_leakage(self):
        overlapping = self.articles.copy()
        overlapping.loc[overlapping.index[-1], "group_id"] = overlapping.iloc[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "Leakage"):
            train_models(overlapping, max_features=200, output_path=None)
        future_in_train = self.articles.copy()
        future_in_train.loc[0, "published_at"] = pd.Timestamp("2030-01-01")
        with self.assertRaisesRegex(ValueError, "Leakage"):
            train_models(future_in_train, max_features=200, output_path=None)

    def test_explanations_reconstruct_log_odds(self):
        title, text = "Shocking rumor", "The secret conspiracy hides a shocking miracle from the public."
        explanation = explain_article(self.bundle, title, text, top_n=100000)
        pipeline = self.bundle["pipeline"]
        classifier = pipeline.named_steps["classifier"]
        actual = float(pipeline.decision_function([data.combine_text(title, text)])[0])
        reconstructed = float(classifier.intercept_[0]) + float(explanation["contribution"].sum())
        self.assertAlmostEqual(reconstructed, actual, places=6)
        probability = predict_article(self.bundle, title, text)["probability_fake"]
        self.assertAlmostEqual(probability, 1 / (1 + np.exp(-actual)), places=8)

    def test_threshold_and_review_region_are_consistent(self):
        title, text = "Report", "Official public council budget approved."
        probability = predict_article(self.bundle, title, text)["probability_fake"]
        reviewed = predict_article(self.bundle, title, text, threshold=probability, review_margin=0.05)
        self.assertEqual(reviewed["decision"], "review")
        low = predict_article(self.bundle, title, text, threshold=0.0, review_margin=0)
        high = predict_article(self.bundle, title, text, threshold=1.0, review_margin=0)
        self.assertEqual(low["decision"], "fake")
        self.assertEqual(high["decision"], "real")

    def test_empty_and_out_of_vocabulary_texts_are_not_confident_decisions(self):
        with self.assertRaises(ValueError):
            predict_article(self.bundle, "", "")
        result = predict_article(self.bundle, "", "несуществующие слова кириллицей")
        self.assertEqual(result["decision"], "review")
        self.assertTrue(result["warnings"])


class AdaptivePolicyTests(unittest.TestCase):
    def test_replay_predicts_before_current_label_feedback(self):
        predictions = sample_articles().iloc[:24].copy()
        predictions["probability_fake"] = np.where(predictions["label"].eq(0), 0.52, 0.99)
        bundle = {
            "threshold": 0.5,
            "test_predictions": predictions,
            "reference_scores": np.linspace(0.1, 0.9, 30),
        }
        with tempfile.TemporaryDirectory() as directory:
            summary = evaluate_replay(bundle, limit=24, output_dir=Path(directory))
            events = pd.read_csv(Path(directory) / "stream_replay.csv")
        self.assertEqual(summary["feedback_source"], "oracle feedback simulation")
        self.assertGreater(summary["threshold_changes"], 0)
        self.assertEqual(events.iloc[0]["adaptive_threshold_before"], 0.5)
        np.testing.assert_allclose(
            events["adaptive_threshold_before"].iloc[1:].to_numpy(),
            events["adaptive_threshold_after"].iloc[:-1].to_numpy(),
        )
        expected = events["probability_fake"].ge(events["adaptive_threshold_before"]).astype(int)
        np.testing.assert_array_equal(events["adaptive_prediction"], expected)
        self.assertGreater(summary["metrics"]["adaptive"]["accuracy"], summary["metrics"]["fixed"]["accuracy"])
        self.assertEqual(summary["events"], 24)

    def test_invalid_feedback_cannot_change_state(self):
        policy = AdaptiveReviewPolicy()
        for probability in (np.nan, np.inf, -np.inf, -0.01, 1.01):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    policy.observe(probability, 0, event_id="invalid")
                with self.assertRaises(ValueError):
                    policy.decide(probability)
        for label in (2, -1, None, "fake"):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    policy.observe(0.5, label, event_id="invalid-label")
        self.assertEqual(policy.stats["feedback_count"], 0)
        self.assertEqual(policy.threshold, 0.5)

    def test_review_region_and_direction(self):
        policy = AdaptiveReviewPolicy(initial_threshold=0.5, review_margin=0.1)
        self.assertEqual(policy.decide(0.2), "real")
        self.assertEqual(policy.decide(0.5), "review")
        self.assertEqual(policy.decide(0.8), "fake")

    def test_feedback_deduplicates_events_and_waits_for_evidence(self):
        policy = AdaptiveReviewPolicy(min_feedback=12)
        first = policy.observe(0.55, 0, event_id="same-article")
        repeated = policy.observe(0.55, 0, event_id="same-article")
        self.assertEqual(first["feedback_count"], repeated["feedback_count"])
        self.assertFalse(repeated["changed"])
        self.assertEqual(policy.threshold, 0.5)
        for index in range(5):
            result = policy.observe(0.55, 0, event_id=f"unique-{index}")
            self.assertFalse(result["changed"])

    def test_threshold_changes_have_bounded_step_and_range(self):
        policy = AdaptiveReviewPolicy(
            initial_threshold=0.5, window_size=30, min_feedback=12,
            step=0.05, min_threshold=0.25, max_threshold=0.75,
        )
        for index in range(120):
            # False positives close enough to the current threshold permit a
            # real improvement within one step, so bounds are tested non-vacuously.
            probability, label = (min(0.99, policy.threshold + 0.02), 0) if index % 2 == 0 else (0.99, 1)
            previous = policy.threshold
            result = policy.observe(probability, label, event_id=f"article-{index}")
            self.assertGreaterEqual(policy.threshold, 0.25)
            self.assertLessEqual(policy.threshold, 0.75)
            self.assertLessEqual(abs(policy.threshold - previous), 0.050000001)
            if result["changed"]:
                self.assertGreaterEqual(result["f1_after"] + 1e-9, result["f1_before"])
        self.assertGreater(policy.stats["threshold_changes"], 0)
        restored = AdaptiveReviewPolicy.from_dict(policy.to_dict())
        self.assertEqual(restored.threshold, policy.threshold)
        self.assertEqual(restored.decide(0.6), policy.decide(0.6))
        count_before = restored.stats["feedback_count"]
        restored.observe(0.99, 1, event_id="article-119")
        self.assertEqual(restored.stats["feedback_count"], count_before)


if __name__ == "__main__":
    unittest.main()

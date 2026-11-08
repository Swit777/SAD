"""Audit contracts: disjoint roles, calibration, consensus, circularity and freeze."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news15.audit import ContaminationIndex, committee_fingerprint, fit_variant, audit_partitions
from news15.consensus import AuditCleaner, policy_statistics, select_consensus_policy, selective_decisions
from news15.model import predict_article, explain_article, save_bundle, load_bundle
from news15.streaming import append_event


def audit_fixture():
    roles = ["train", "tuning", "calibration_members", "calibration_fusion", "policy", "internal_reused_test"]
    rows = []
    for r, role in enumerate(roles):
        for i in range(40):
            label = i % 2
            body = ("official council budget verified report documents " if label == 0 else "fabricated conspiracy rumor magic hoax false ")
            body += f" detailed public discussion record {r} item {i}."
            if role == "policy":
                body += " policyonlysentinel"
            rows.append({
                "id": f"{role}-{i}", "title": f"Unique {role} event {i}",
                "text": body, "label": label, "subject": "test",
                "published_at": pd.Timestamp("2017-01-01") + pd.Timedelta(days=r*30+i//2),
                "group_id": f"{role}-{i}", "role": role,
            })
    return pd.DataFrame(rows)


class ConsensusPolicyTests(unittest.TestCase):
    def test_unanimous_wrong_predictions_do_not_satisfy_quality_constraints(self):
        y = np.tile([0, 1], 200)
        p = np.where(y == 0, .99, .01)
        members = np.repeat(p[:, None], 3, axis=1)
        policy, table = select_consensus_policy(y, p, members)
        self.assertFalse(policy["enabled"])
        self.assertFalse(table.feasible.any())
        _, decisions, agreement = selective_decisions(p, members, policy)
        self.assertTrue((decisions == "review").all())
        self.assertTrue((agreement == 1).all())

    def test_policy_requires_per_class_evidence_and_correct_consensus(self):
        y = np.tile([0, 1], 200)
        p = np.where(y == 0, .01, .99)
        members = np.repeat(p[:, None], 3, axis=1)
        policy, _ = select_consensus_policy(y, p, members)
        self.assertTrue(policy["enabled"])
        stats = policy_statistics(y, p, members, policy)
        self.assertGreaterEqual(stats["fake_precision_wilson_lower"], .9)
        self.assertGreaterEqual(stats["real_precision_wilson_lower"], .9)
        policy = {"threshold": .5, "margin": .1, "consensus": 1., "enabled": True}
        _, decisions, _ = selective_decisions([.9], [[.9, .9, .1]], policy)
        self.assertEqual(decisions[0], "review")

    def test_source_marker_removal_preserves_negation(self):
        value = AuditCleaner().fit_transform(["WASHINGTON (Reuters) - Evidence has not changed. Featured image Getty."])[0]
        self.assertIn("not changed", value)
        self.assertNotIn("reuters", value)
        self.assertNotIn("getty", value)

    def test_contamination_checks_near_headlines_without_using_labels(self):
        index = ContaminationIndex()
        frame = pd.DataFrame({"title": ["Council approves the annual school budget after public meeting"], "text": ["A detailed body about the annual school budget."]})
        index.add(frame)
        self.assertIsNotNone(index.match("BREAKING: Council approves the annual school budget after public meeting"))
        self.assertIsNone(index.match("Completely unrelated science experiment on ocean temperatures"))

    def test_persistent_logs_do_not_store_raw_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            append_event({"id": "1", "title": "A person's name", "text": "private body", "probability": .4}, path)
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("title", row)
            self.assertNotIn("text", row)
            self.assertEqual(row["id"], "1")


class FrozenCommitteeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        frame = audit_fixture()
        cls.committee, _, _, cls.background, _ = fit_variant(frame, True, Path(cls.temp.name))
        from sklearn.pipeline import Pipeline
        member = cls.committee.members[0]["estimator"]
        cls.bundle = {
            "schema_version": 2, "consensus": cls.committee, "background_mean": cls.background,
            "pipeline": Pipeline([("clean", cls.committee.cleaner), ("tfidf", cls.committee.vectorizer), ("classifier", member)]),
            "metadata": {"release": {"automatic_user_verdicts_enabled": False}},
        }

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_vocabulary_never_sees_policy_validation(self):
        self.assertNotIn("policyonlysentinel", self.committee.vectorizer.vocabulary_)

    def test_shap_is_additive_for_component_not_claimed_for_consensus(self):
        explanation = explain_article(self.bundle, "Budget", "Official council budget documents have not changed", top_n=10000)
        self.assertAlmostEqual(explanation.attrs["intercept"]+explanation.contribution.sum(), explanation.attrs["logit"], places=10)
        self.assertIn("NOT final consensus", explanation.attrs["explanation_target"])

    def test_external_predictions_cannot_mutate_the_frozen_policy(self):
        fingerprint = committee_fingerprint(self.committee)
        self.committee.predict_proba(["A completely new raw external headline"])
        self.assertEqual(fingerprint, committee_fingerprint(self.committee))
        self.committee.policy = {**self.committee.policy, "threshold": self.committee.policy["threshold"]+.01}
        self.assertNotEqual(fingerprint, committee_fingerprint(self.committee))
        self.committee.policy["threshold"] -= .01

    def test_research_release_gate_overrides_confident_automatic_verdict(self):
        result = predict_article(self.bundle, "Official report", "The official council budget documents support the verified public report")
        self.assertEqual(result["decision"], "review")
        self.assertIsNotNone(result["review_reason"])
        self.assertEqual(len(result["members"]), 3)

    def test_version_two_round_trip_preserves_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            save_bundle(self.bundle, path)
            restored = load_bundle(path)
            expected = self.committee.predict_proba(["Official council budget"])
            np.testing.assert_allclose(expected, restored["consensus"].predict_proba(["Official council budget"]))


if __name__ == "__main__":
    unittest.main()

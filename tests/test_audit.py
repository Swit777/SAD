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
from news15.training import learning_partitions, measured_metrics
from news15.model import committee_for_input


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


class ImprovedEnglishContracts(unittest.TestCase):
    @staticmethod
    def grouped_fixture():
        return pd.DataFrame([{
            'id':f'row-{i}','group_id':f'group-{i}',
            'title':f'{"Council" if i%2 else "Market"} report topic {i} unique item',
            'text':'Original body for this article.','label':i%2,
            'eligibility':'legacy_train' if i<400 else 'new_candidate_unassigned',
            'published_at':pd.NaT,'external_overlap':False,
        } for i in range(800)])

    def test_exact_headline_copies_stay_in_one_role(self):
        frame=self.grouped_fixture()
        frame.loc[410,'title']=frame.loc[10,'title']
        result=learning_partitions(frame)
        self.assertEqual(result.loc[410,'learning_group'],result.loc[10,'learning_group'])
        self.assertEqual(result.loc[410,'role'],result.loc[10,'role'])
        self.assertEqual(result.groupby('learning_group').role.nunique().max(),1)
        self.assertTrue(result.published_at.isna().all())

    def test_reserved_copy_reserves_the_whole_headline_group(self):
        frame=self.grouped_fixture()
        frame.loc[10,'eligibility']='reserved_legacy_test'
        frame.loc[410,'title']=frame.loc[10,'title']
        result=learning_partitions(frame)
        self.assertEqual(result.loc[10,'role'],'prior_reserved')
        self.assertEqual(result.loc[410,'role'],'prior_reserved')

    def test_conflicting_group_is_quarantined_without_voting(self):
        frame=self.grouped_fixture()
        frame.loc[411,'title']=frame.loc[10,'title']
        result=learning_partitions(frame)
        self.assertTrue(result.loc[[10,411],'role'].eq('quarantine_conflicting_labels').all())

    def test_external_text_exclusion_propagates_to_all_copies(self):
        frame=self.grouped_fixture();frame.loc[10,'external_overlap']=True
        frame.loc[410,'title']=frame.loc[10,'title']
        result=learning_partitions(frame)
        self.assertTrue(result.loc[[10,410],'role'].eq('reserved_new_external').all())

    def test_short_and_long_inputs_use_their_own_calibrator(self):
        full,short=object(),object()
        bundle={'consensus':full,'headline_consensus':short}
        self.assertIs(committee_for_input(bundle,''),short)
        self.assertIs(committee_for_input(bundle,'short manually written text'),short)
        self.assertIs(committee_for_input(bundle,'word '*40),full)
        self.assertIs(committee_for_input({'consensus':full},''),full)

    def test_all_fake_predictions_expose_false_accusations_despite_recall(self):
        metrics=measured_metrics([0,1,0,1],[.99,.99,.99,.99],.5,['a','a','b','b'],repetitions=10)
        self.assertEqual(metrics['recall'],1.)
        self.assertEqual(metrics['false_fake_rate'],1.)
        self.assertLess(metrics['f1_macro'],.4)

    def test_headline_prose_and_combined_length_are_routed_consistently(self):
        full,short,headline=object(),object(),object()
        bundle={'consensus':full,'short_consensus':short,'headline_consensus':headline}
        self.assertIs(committee_for_input(bundle,'','a headline'),headline)
        self.assertIs(committee_for_input(bundle,'a short statement','a headline'),short)
        self.assertIs(committee_for_input(bundle,'word '*25,'title '*20),full)

    def test_fusion_uses_each_member_logit_and_detects_configuration_changes(self):
        from news15.consensus import CalibratedConsensus
        committee=CalibratedConsensus(None,None,[],None,{},fusion_mode='member_logits',fusion_clip=.01)
        values=committee.fusion_features(np.asarray([[0.,.5,1.]]))
        self.assertEqual(values.shape,(1,3))
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(values[0,1],0.)
        np.testing.assert_allclose(values[0,0],-values[0,2])

    def test_conservative_threshold_cannot_win_by_predicting_everything_real(self):
        from news15.training import refined_policy
        y=np.tile([0,1],100)
        p=np.where(y==0,.40,.65)
        members=np.repeat(p[:,None],3,axis=1)
        policy,rows=refined_policy(y,p,members)
        self.assertGreater(policy['threshold'],.40)
        self.assertLessEqual(policy['threshold'],.65)

    def test_float32_word_character_shap_is_additive_and_weights_are_fingerprinted(self):
        from sklearn.pipeline import FeatureUnion
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier,LogisticRegression
        from news15.consensus import CalibratedConsensus,raw_score
        texts=['official council report approved budget','secret rumor miracle hoax']*20
        labels=np.tile([0,1],20)
        cleaner=AuditCleaner(True,max_chars=1600)
        vectorizer=FeatureUnion([
            ('word',TfidfVectorizer(dtype=np.float32)),
            ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,4),dtype=np.float32)),
        ],transformer_weights={'word':1.,'char':.4})
        X=vectorizer.fit_transform(cleaner.transform(texts))
        estimator=SGDClassifier(loss='log_loss',random_state=42).fit(X,labels)
        calibration=LogisticRegression().fit(raw_score(estimator,X).reshape(-1,1),labels)
        member={'name':'primary','estimator':estimator,'calibrator':calibration}
        committee=CalibratedConsensus(cleaner,vectorizer,[member]*3,calibration,
            {'threshold':.5,'margin':.1,'consensus':1.,'enabled':False})
        phi,base,logits=committee.component_shap(texts[:2],np.asarray(X.mean(axis=0)).reshape(-1))
        np.testing.assert_allclose(phi.sum(axis=1)+base,logits,atol=1e-10,rtol=0)
        initial=committee_fingerprint(committee)
        vectorizer.transformer_weights['char']=.5
        self.assertNotEqual(initial,committee_fingerprint(committee))

    def test_stacked_explanation_reconstructs_final_probability_and_marks_absence(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier,LogisticRegression
        from scipy.special import expit
        from news15.consensus import CalibratedConsensus,raw_score
        texts=['official council approved budget report','secret miracle shocking fabricated rumor']*20
        labels=np.tile([0,1],20)
        cleaner=AuditCleaner();vectorizer=TfidfVectorizer(dtype=np.float32)
        X=vectorizer.fit_transform(cleaner.transform(texts))
        classifier=SGDClassifier(loss='log_loss',random_state=42).fit(X,labels)
        calibrator=LogisticRegression().fit(raw_score(classifier,X).reshape(-1,1),labels)
        members=[{'name':str(i),'estimator':classifier,'calibrator':calibrator} for i in range(3)]
        committee=CalibratedConsensus(cleaner,vectorizer,members,LogisticRegression(),
            {'threshold':.5,'margin':.1,'consensus':1.,'enabled':False},fusion_mode='member_logits',fusion_clip=.01)
        probabilities=np.repeat(calibrator.predict_proba(raw_score(classifier,X).reshape(-1,1))[:,1,None],3,axis=1)
        committee.fusion.fit(committee.fusion_features(probabilities),labels)
        bundle={'schema_version':2,'consensus':committee,'headline_consensus':committee,'short_consensus':committee,
                'pipeline':Pipeline([('clean',cleaner),('tfidf',vectorizer),('classifier',classifier)]),
                'metadata':{'release':{'automatic_user_verdicts_enabled':False}},'background_mean':np.asarray(X.mean(axis=0)).reshape(-1)}
        result=predict_article(bundle,'Official council report','The council approved the budget.')
        total=result['fusion_explanation']['intercept']+sum(row['fusion_logit_contribution'] for row in result['members'])
        self.assertAlmostEqual(float(expit(total)),result['probability_fake'],places=12)
        self.assertEqual(result['decision'],'review')
        explanation=explain_article(bundle,'Official council report','The council approved the budget.',top_n=1000)
        self.assertTrue(explanation.present_in_text.any())
        self.assertTrue((~explanation.present_in_text).any())


if __name__ == "__main__":
    unittest.main()

"""Frozen calibrated committee and auditable selective decision policy."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin

from .data import clean_text

# Declared before external validation. These are source/formatting markers,
# not terms chosen because they have a large SHAP value or ANOVA F statistic.
BOILERPLATE_TOKENS = (
    "reuters", "thomson", "reporting", "editing", "copyright", "reserved",
    "getty", "featured", "image", "images", "pic", "twitter", "facebook",
    "subscribe", "newsletter", "click", "watch", "video", "videos", "via",
    "follow", "share", "read", "com", "www", "http", "https",
)


class AuditCleaner(TransformerMixin, BaseEstimator):
    def __init__(self, remove_boilerplate=True):
        self.remove_boilerplate = remove_boilerplate

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for text in X:
            value = str(text or "")
            if self.remove_boilerplate:
                value = re.sub(r"(?is)\b(?:reporting by|editing by|featured image|photo credit|image credit)\b.*$", " ", value)
                value = re.sub(r"(?im)^\s*(?:source|follow us|subscribe|all rights reserved)\s*:?.*$", " ", value)
            value = clean_text(value)
            if self.remove_boilerplate:
                value = re.sub(r"\b(?:" + "|".join(BOILERPLATE_TOKENS) + r")\b", " ", value)
            result.append(re.sub(r"\s+", " ", value).strip())
        return np.asarray(result, dtype=object)


def raw_score(estimator, X):
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float)
    logs = estimator.predict_log_proba(X)
    return logs[:, 1] - logs[:, 0]


def wilson_lower(correct, total, z=1.96):
    if total == 0:
        return 0.0
    p = correct / total
    return float((p + z*z/(2*total) - z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total))


def selective_decisions(probabilities, member_probabilities, policy):
    probabilities = np.asarray(probabilities, dtype=float)
    members = np.asarray(member_probabilities, dtype=float)
    if members.ndim != 2 or len(members) != len(probabilities):
        raise ValueError("Member scores must be an n_articles by n_members matrix.")
    threshold = float(policy["threshold"])
    predicted = (probabilities >= threshold).astype(int)
    votes = members >= 0.5
    agreement = np.where(predicted == 1, votes.mean(axis=1), (~votes).mean(axis=1))
    accepted = (
        (np.abs(probabilities - threshold) >= float(policy["margin"]))
        & (agreement >= float(policy["consensus"]) - 1e-9)
        & bool(policy.get("enabled", True))
    )
    decisions = np.where(accepted, np.where(predicted == 1, "fake", "real"), "review")
    return predicted, decisions, agreement


def policy_statistics(labels, probabilities, member_probabilities, policy):
    labels = np.asarray(labels, dtype=int)
    predicted, decisions, agreement = selective_decisions(probabilities, member_probabilities, policy)
    accepted = decisions != "review"
    fake = accepted & (predicted == 1)
    real = accepted & (predicted == 0)
    fp = int((fake & (labels == 0)).sum())
    fn = int((real & (labels == 1)).sum())
    return {
        "coverage": float(accepted.mean()), "review_fraction": float((~accepted).mean()),
        "automatic_fake_n": int(fake.sum()), "automatic_real_n": int(real.sum()),
        "automatic_fake_precision": float((labels[fake] == 1).mean()) if fake.any() else None,
        "automatic_real_precision": float((labels[real] == 0).mean()) if real.any() else None,
        "fake_precision_wilson_lower": wilson_lower(int((fake & (labels == 1)).sum()), int(fake.sum())),
        "real_precision_wilson_lower": wilson_lower(int((real & (labels == 0)).sum()), int(real.sum())),
        "automatic_errors": fp + fn, "false_fake": fp, "false_real": fn,
        "automatic_accuracy": float((labels[accepted] == predicted[accepted]).mean()) if accepted.any() else None,
        "cost_per_article": float((5 * fp + fn + 0.15 * (~accepted).sum()) / len(labels)),
        "mean_agreement": float(agreement.mean()),
    }


def select_consensus_policy(labels, probabilities, member_probabilities):
    """Select using policy-validation only; constraints are fixed in the protocol."""
    rows = []
    for threshold in np.linspace(0.25, 0.75, 11):
        for margin in (0.0, 0.05, 0.10, 0.15, 0.20):
            for consensus in (2/3, 1.0):
                policy = {"threshold": float(threshold), "margin": margin, "consensus": consensus, "enabled": True}
                stats = policy_statistics(labels, probabilities, member_probabilities, policy)
                feasible = (
                    stats["coverage"] >= 0.50 and stats["automatic_fake_n"] >= 30
                    and stats["automatic_real_n"] >= 30
                    and stats["fake_precision_wilson_lower"] >= 0.90
                    and stats["real_precision_wilson_lower"] >= 0.90
                )
                rows.append({**policy, **stats, "feasible": feasible})
    table = pd.DataFrame(rows)
    choices = table[table["feasible"]].sort_values(
        ["cost_per_article", "coverage", "margin", "threshold"], ascending=[True, False, True, True]
    )
    if choices.empty:
        policy = {"threshold": 0.5, "margin": 0.1, "consensus": 1.0, "enabled": False}
    else:
        best = choices.iloc[0]
        policy = {key: float(best[key]) for key in ("threshold", "margin", "consensus")}
        policy["enabled"] = True
    return policy, table


class CalibratedConsensus:
    def __init__(self, cleaner, vectorizer, members, fusion, policy):
        self.cleaner = cleaner
        self.vectorizer = vectorizer
        self.members = members
        self.fusion = fusion
        self.policy = policy

    def vectorize(self, texts):
        return self.vectorizer.transform(self.cleaner.transform(texts))

    def details_from_matrix(self, X):
        raw = np.column_stack([raw_score(member["estimator"], X) for member in self.members])
        calibrated = np.column_stack([
            member["calibrator"].predict_proba(raw[:, i:i+1])[:, 1]
            for i, member in enumerate(self.members)
        ])
        mean_score = calibrated.mean(axis=1)
        fusion_input = logit(np.clip(mean_score, 1e-6, 1-1e-6)).reshape(-1, 1)
        probabilities = self.fusion.predict_proba(fusion_input)[:, 1]
        predicted, decisions, agreement = selective_decisions(probabilities, calibrated, self.policy)
        return {
            "probabilities": probabilities, "member_probabilities": calibrated,
            "uncalibrated_member_probabilities": expit(raw),
            "prediction": predicted, "decisions": decisions, "agreement": agreement,
        }

    def details(self, texts):
        return self.details_from_matrix(self.vectorize(texts))

    def predict_proba(self, texts):
        p = self.details(texts)["probabilities"]
        return np.column_stack([1-p, p])

    def component_shap(self, texts, background_mean):
        """Exact independent/interventional linear SHAP for component 0 logit.

        This explains one calibrated member, not the nonlinear final consensus.
        The background is fixed using training examples, never evaluation labels.
        """
        X = self.vectorize(texts)
        primary = self.members[0]
        scale = float(primary["calibrator"].coef_[0, 0])
        coefficients = primary["estimator"].coef_[0] * scale
        intercept = float(primary["calibrator"].intercept_[0] + primary["estimator"].intercept_[0] * scale)
        mean = np.asarray(background_mean).reshape(-1)
        phi = (X.toarray() - mean) * coefficients
        base = float(intercept + mean @ coefficients)
        logits = np.asarray(X @ coefficients).reshape(-1) + intercept
        return phi, base, logits

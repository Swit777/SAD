"""Versioned methodological audit with disjoint tuning and locked external data."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import re
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from .consensus import (
    AuditCleaner, BOILERPLATE_TOKENS, CalibratedConsensus, policy_statistics,
    raw_score, select_consensus_policy, selective_decisions,
)
from .data import article_texts, clean_text, file_sha256, load_articles
from .model import evaluate_predictions, load_bundle, save_bundle
from .paths import ARTIFACTS_DIR, MODEL_PATH, PROJECT_DIR, REPORTS_DIR

AUDIT_DIR = REPORTS_DIR / "audit_v2"
PROTOCOL_PATH = PROJECT_DIR / "protocols" / "v2.json"
SEED = 42


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def title_key(value):
    value = clean_text(value)
    value = re.sub(r"^(?:update\s*\d*|corrected|exclusive|breaking)\s*[-:–—]*\s*", "", value)
    return " ".join(re.findall(r"[a-z0-9]+", value))


def shingles(value, width):
    words = title_key(value).split()
    return frozenset(" ".join(words[i:i+width]) for i in range(max(0, len(words)-width+1)))


class ContaminationIndex:
    """Bounded exact + shingle near-duplicate screen; not an exhaustive proof."""
    def __init__(self):
        self.exact_titles = set()
        self.exact_bodies = set()
        self.postings = defaultdict(list)
        self.documents = []

    def add(self, frame):
        for row in frame.itertuples():
            title = title_key(row.title)
            body = clean_text(row.text)
            if title:
                self.exact_titles.add(title)
            if body:
                self.exact_bodies.add(hashlib.sha256(body.encode()).hexdigest())
            grams = shingles(title, 3)
            index = len(self.documents)
            self.documents.append(grams)
            for gram in grams:
                if len(self.postings[gram]) < 201:
                    self.postings[gram].append(index)

    def match(self, title, text=""):
        key = title_key(title)
        if key and key in self.exact_titles:
            return "exact_headline"
        if text and hashlib.sha256(clean_text(text).encode()).hexdigest() in self.exact_bodies:
            return "exact_body"
        grams = shingles(key, 3)
        if len(grams) < 2:
            return None
        candidates = set()
        for gram in grams:
            posting = self.postings.get(gram, [])
            if len(posting) <= 200:
                candidates.update(posting)
        for candidate in sorted(candidates)[:1000]:
            other = self.documents[candidate]
            union = len(grams | other)
            if union and len(grams & other) / union >= 0.80:
                return "near_headline_jaccard"
        return None


def audit_partitions(articles):
    """Refine validation into four independent chronological roles."""
    frame = articles.copy()
    validation = frame[frame.split == "validation"].sort_values("published_at")
    dates = validation.published_at.dt.normalize()
    boundaries = [dates.iloc[int(len(dates)*q)] for q in (0.4, 0.6, 0.8)]
    frame["role"] = frame["split"].replace({"test": "internal_reused_test"})
    val_roles = np.select(
        [dates < boundaries[0], dates < boundaries[1], dates < boundaries[2]],
        ["tuning", "calibration_members", "calibration_fusion"], default="policy",
    )
    frame.loc[validation.index, "role"] = val_roles
    roles = ["train", "tuning", "calibration_members", "calibration_fusion", "policy", "internal_reused_test"]
    index = ContaminationIndex()
    kept = []
    removed = []
    for role in roles:
        part = frame[frame.role == role].sort_values(["published_at", "id"])
        accepted = []
        for row in part.itertuples():
            reason = index.match(row.title, row.text) if role != "train" else None
            if reason:
                removed.append({"id": row.id, "role": role, "reason": reason})
            else:
                accepted.append(row.id)
        part = part[part.id.isin(accepted)].copy()
        if part.label.nunique() != 2:
            raise ValueError(f"Both classes must survive contamination screening in {role}.")
        kept.append(part)
        index.add(part)
    result = pd.concat(kept).reset_index(drop=True)
    details = {
        role: {
            "n": len(part), "real": int((part.label == 0).sum()), "fake": int((part.label == 1).sum()),
            "start": part.published_at.min().isoformat(), "end": part.published_at.max().isoformat(),
        } for role, part in result.groupby("role")
    }
    return result, pd.DataFrame(removed), details


def classification_with_interval(labels, probabilities, threshold, repetitions=200):
    labels = np.asarray(labels, int)
    probabilities = np.asarray(probabilities, float)
    result = evaluate_predictions(labels, probabilities, threshold)
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    values = []
    predicted = probabilities >= threshold
    for _ in range(repetitions):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups if len(group)])
        values.append(f1_score(labels[sample], predicted[sample], labels=[0, 1], average="macro", zero_division=0))
    result["macro_f1_ci95"] = np.quantile(values, [.025, .975]).tolist()
    result["ci_method"] = "200 label-stratified article bootstraps; no publisher clustering available."
    return result


def calibration_table(labels, probabilities, bins=10):
    labels = np.asarray(labels)
    p = np.asarray(probabilities)
    rows = []
    for i in range(bins):
        lower, upper = i/bins, (i+1)/bins
        mask = (p >= lower) & ((p <= upper) if i == bins-1 else (p < upper))
        if mask.any():
            rows.append({"bin_lower": lower, "bin_upper": upper, "n": int(mask.sum()), "mean_score": float(p[mask].mean()), "fake_fraction": float(labels[mask].mean())})
    ece = sum(row["n"]/len(labels)*abs(row["mean_score"]-row["fake_fraction"]) for row in rows)
    return pd.DataFrame(rows), float(ece)


def fit_variant(frame, remove_boilerplate, directory):
    directory.mkdir(parents=True, exist_ok=True)
    parts = {role: group for role, group in frame.groupby("role")}
    cleaner = AuditCleaner(remove_boilerplate)
    vectorizer = TfidfVectorizer(
        max_features=20000, min_df=3, max_df=.98, ngram_range=(1, 2),
        sublinear_tf=True, stop_words=sorted(set(ENGLISH_STOP_WORDS)-{"no", "not", "nor"}),
        strip_accents="unicode",
    )
    print(f"Features: {'after' if remove_boilerplate else 'before'} removal", flush=True)
    matrices = {}
    train_texts = article_texts(parts["train"])
    matrices["train"] = vectorizer.fit_transform(cleaner.fit_transform(train_texts))
    for role in parts:
        if role != "train":
            matrices[role] = vectorizer.transform(cleaner.transform(article_texts(parts[role])))
    y = parts["train"].label.to_numpy(int)
    candidates = [
        ("logistic", {"C": c}, LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=SEED))
        for c in (.25, 1., 4.)
    ] + [
        ("sgd", {"alpha": alpha}, SGDClassifier(loss="log_loss", alpha=alpha, class_weight="balanced", max_iter=1500, tol=1e-4, random_state=SEED))
        for alpha in (.0001, .001)
    ] + [
        ("complement_nb", {"alpha": alpha}, ComplementNB(alpha=alpha)) for alpha in (.5, 1.)
    ]
    best = {}
    tuning_rows = []
    for family, params, estimator in candidates:
        estimator.fit(matrices["train"], y)
        probability = estimator.predict_proba(matrices["tuning"])[:, 1]
        metric = f1_score(parts["tuning"].label, probability >= .5, average="macro")
        tuning_rows.append({"family": family, "parameters": json.dumps(params), "tuning_macro_f1": float(metric)})
        if family not in best or metric > best[family][0]:
            best[family] = (metric, estimator, params)
    pd.DataFrame(tuning_rows).to_csv(directory / "model_tuning.csv", index=False)
    members = []
    for name in ("logistic", "sgd", "complement_nb"):
        estimator = best[name][1]
        calibrator = LogisticRegression(C=100., solver="lbfgs", max_iter=1000)
        scores = raw_score(estimator, matrices["calibration_members"])
        calibrator.fit(scores.reshape(-1, 1), parts["calibration_members"].label)
        members.append({"name": name, "estimator": estimator, "calibrator": calibrator, "parameters": best[name][2]})
    member_fusion_scores = np.column_stack([
        member["calibrator"].predict_proba(raw_score(member["estimator"], matrices["calibration_fusion"]).reshape(-1, 1))[:, 1]
        for member in members
    ])
    fusion = LogisticRegression(C=100., solver="lbfgs", max_iter=1000)
    fusion.fit(logit(np.clip(member_fusion_scores.mean(axis=1), 1e-6, 1-1e-6)).reshape(-1, 1), parts["calibration_fusion"].label)
    committee = CalibratedConsensus(cleaner, vectorizer, members, fusion, {"threshold": .5, "margin": .1, "consensus": 1., "enabled": True})
    detail = committee.details_from_matrix(matrices["policy"])
    policy, grid = select_consensus_policy(parts["policy"].label, detail["probabilities"], detail["member_probabilities"])
    committee.policy = policy
    grid.to_csv(directory / "policy_grid.csv", index=False)
    background = np.asarray(matrices["train"].mean(axis=0)).reshape(-1)
    return committee, matrices, parts, background, tuning_rows


def committee_fingerprint(committee):
    digest = hashlib.sha256()
    digest.update(json.dumps({key: int(value) for key, value in committee.vectorizer.vocabulary_.items()}, sort_keys=True).encode())
    digest.update(np.asarray(committee.vectorizer.idf_).tobytes())
    for member in committee.members:
        for obj in (member["estimator"], member["calibrator"]):
            for attribute in ("coef_", "intercept_", "feature_log_prob_"):
                if hasattr(obj, attribute):
                    digest.update(np.asarray(getattr(obj, attribute)).tobytes())
    digest.update(committee.fusion.coef_.tobytes())
    digest.update(committee.fusion.intercept_.tobytes())
    digest.update(json.dumps(committee.policy, sort_keys=True).encode())
    return digest.hexdigest()


def fetch_external(protocol, development, directory):
    """Raw titles only. Do not use their labels for feature/model/policy tuning."""
    raw_dir = PROJECT_DIR / "data" / "raw" / "fakenewsnet"
    raw_dir.mkdir(parents=True, exist_ok=True)
    commit = protocol["external_commit"]
    manifest = []
    frames = []
    for source in ("politifact", "gossipcop"):
        for kind, label in (("real", 0), ("fake", 1)):
            name = f"{source}_{kind}.csv"
            url = f"https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/{commit}/dataset/{name}"
            path = raw_dir / name
            if not path.exists():
                response = requests.get(url, timeout=(15, 90))
                response.raise_for_status()
                path.write_bytes(response.content)
            raw = pd.read_csv(path, keep_default_na=False)
            if not {"id", "title", "news_url"}.issubset(raw):
                raise ValueError(f"Unexpected raw external CSV schema: {name}")
            raw["label"] = label
            raw["external_source"] = source
            raw["text"] = ""
            raw["raw_file"] = name
            raw["title_key"] = raw.title.map(title_key)
            frames.append(raw[["id", "title", "text", "label", "external_source", "raw_file", "title_key"]])
            manifest.append({"file": name, "url": url, "sha256": file_sha256(path), "rows": len(raw)})
    frame = pd.concat(frames, ignore_index=True)
    original_n = len(frame)
    contradictory_ids = frame.groupby("id").label.nunique()
    conflicting_ids = set(contradictory_ids[contradictory_ids > 1].index)
    frame = frame[~frame.id.isin(conflicting_ids)].drop_duplicates("id").copy()
    frame = frame[frame.title_key.str.len() >= 10].copy()
    contradictory = frame.groupby("title_key").label.nunique()
    conflicts = set(contradictory[contradictory > 1].index)
    frame = frame[~frame.title_key.isin(conflicts)]
    frame = frame.drop_duplicates("title_key").copy()
    index = ContaminationIndex()
    index.add(development)
    reasons = [index.match(row.title) for row in frame.itertuples()]
    excluded = frame.loc[np.asarray([reason is not None for reason in reasons])].copy()
    excluded["exclusion"] = [reason for reason in reasons if reason is not None]
    excluded[["id", "external_source", "exclusion"]].to_csv(directory / "external_contamination_exclusions.csv", index=False)
    frame = frame.loc[np.asarray([reason is None for reason in reasons])].reset_index(drop=True)
    manifest_report = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
        "raw_rows": original_n, "evaluated_rows": len(frame), "conflicting_title_groups": len(conflicts),
        "conflicting_id_groups": len(conflicting_ids),
        "internal_overlap_exclusions": len(excluded), "files": manifest,
        "unit": "raw headlines only; titles passed unchanged to the frozen preprocessing pipeline",
        "limitations": [
            "No full article text is provided in the public CSV; this tests headline transfer, not full-text transfer.",
            "Exact and bounded shingle screens reduce contamination but cannot prove complete independence.",
            "No external label is used to fit, select, calibrate, remove features or tune thresholds.",
        ],
    }
    write_json(directory / "external_manifest.json", manifest_report)
    return frame, manifest_report


def agreement_report(labels, details, names, directory, prefix):
    labels = np.asarray(labels, int)
    votes = details["member_probabilities"] >= .5
    rows = []
    for i, first in enumerate(names):
        for j, second in enumerate(names):
            kappa = float(cohen_kappa_score(votes[:, i], votes[:, j])) if i != j else 1.
            rows.append({
                "first": first, "second": second,
                "prediction_agreement": float((votes[:, i] == votes[:, j]).mean()),
                "cohen_kappa": kappa if np.isfinite(kappa) else None,
                "both_wrong_fraction": float(((votes[:, i] != labels) & (votes[:, j] != labels)).mean()),
            })
    pd.DataFrame(rows).to_csv(directory / f"{prefix}_agreement.csv", index=False)
    unanimous = votes.all(axis=1) | (~votes).all(axis=1)
    return {
        "pairwise": rows, "unanimous_n": int(unanimous.sum()),
        "unanimous_wrong_fraction": float((votes[unanimous, 0] != labels[unanimous]).mean()) if unanimous.any() else None,
        "interpretation": "Correlated models trained on one corpus are not independent fact-checkers.",
    }


def evaluate_committee(committee, frame, directory, name, headline_only=False):
    texts = frame.title.astype(str).tolist() if headline_only else article_texts(frame)
    details = committee.details(texts)
    p = details["probabilities"]
    metrics = classification_with_interval(frame.label, p, committee.policy["threshold"])
    selective = policy_statistics(frame.label, p, details["member_probabilities"], committee.policy)
    reliability, ece = calibration_table(frame.label, p)
    reliability.to_csv(directory / f"{name}_calibration.csv", index=False)
    metrics["ece_10_bins"] = ece
    primary_raw = details["uncalibrated_member_probabilities"][:, 0]
    metrics["primary_uncalibrated_brier"] = float(np.mean((primary_raw-frame.label.to_numpy())**2))
    prediction = pd.DataFrame({
        "id": frame.id.astype(str), "label": frame.label.astype(int),
        "probability_fake": p, "prediction": details["prediction"],
        "decision": details["decisions"], "agreement": details["agreement"],
    })
    for i, member in enumerate(committee.members):
        prediction[f"member_{member['name']}"] = details["member_probabilities"][:, i]
    prediction.to_csv(directory / f"{name}_predictions.csv", index=False)
    agreement = agreement_report(frame.label, details, [member["name"] for member in committee.members], directory, name)
    return {"metrics": metrics, "selective_policy": selective, "agreement": agreement}, details


def feature_and_circularity_audit(committee, matrices, parts, background, directory, permutations):
    X = matrices["train"]
    y = parts["train"].label.to_numpy(int)
    source = np.asarray([bool(re.search(r"\breuters\b", str(text), re.I)) for text in parts["train"].text], int)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_label, p_label = f_classif(X, y)
        f_source, _ = f_classif(X, source)
    f_label = np.nan_to_num(f_label, nan=0., posinf=1e10)
    f_source = np.nan_to_num(f_source, nan=0., posinf=1e10)
    p_label = np.nan_to_num(p_label, nan=1.)
    order = np.argsort(p_label)
    adjusted = p_label[order] * len(p_label) / np.arange(1, len(p_label)+1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    q_values = np.empty_like(p_label)
    q_values[order] = adjusted
    primary = committee.members[0]
    scale = float(primary["calibrator"].coef_[0, 0])
    coef = primary["estimator"].coef_[0] * scale
    sample = parts["train"].sample(min(256, len(y)), random_state=SEED)
    phi, base, logits = committee.component_shap(article_texts(sample), background)
    exact_error = float(np.max(np.abs(base + phi.sum(axis=1) - logits)))
    global_importance = np.abs(phi).mean(axis=0)
    vocabulary = committee.vectorizer.get_feature_names_out()
    table = pd.DataFrame({
        "feature": vocabulary, "anova_F_label": f_label, "anova_p_label": p_label,
        "anova_q_BH": q_values,
        "anova_F_source_proxy": f_source, "global_mean_abs_component_SHAP": global_importance,
        "calibrated_component_coefficient": coef,
    })
    table.sort_values("global_mean_abs_component_SHAP", ascending=False).to_csv(directory / "global_xai_anova.csv", index=False)
    # A valid train-only ANOVA selection negative control, refitted for each shuffle.
    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(len(y), min(1600, len(y)), replace=False)
    holdout = parts["tuning"].sample(min(600, len(parts["tuning"])), random_state=SEED)
    heldout_X = committee.vectorize(article_texts(holdout))
    observed = None
    controls = []
    for permutation in range(permutations + 1):
        target = y[train_idx].copy()
        if permutation:
            rng.shuffle(target)
        selector = SelectKBest(f_classif, k=min(500, X.shape[1]))
        reduced = selector.fit_transform(X[train_idx], target)
        estimator = LogisticRegression(C=1., class_weight="balanced", solver="liblinear", max_iter=700, random_state=SEED)
        estimator.fit(reduced, target)
        score = estimator.predict_proba(selector.transform(heldout_X))[:, 1]
        auc = float(roc_auc_score(holdout.label, score))
        # Even a model fitted on shuffled labels has nonzero linear SHAP values.
        selected_mean = np.asarray(reduced.mean(axis=0)).reshape(-1)
        small = selector.transform(heldout_X[:40]).toarray()
        shap_magnitude = float(np.abs((small-selected_mean)*estimator.coef_[0]).mean())
        row = {
            "permutation": permutation, "labels_shuffled": bool(permutation),
            "heldout_auc": auc, "heldout_macro_f1": float(f1_score(holdout.label, score >= .5, average="macro")),
            "nonzero_SHAP_magnitude": shap_magnitude,
            "max_selected_train_ANOVA_F": float(np.nanmax(selector.scores_)),
        }
        controls.append(row)
        if permutation == 0:
            observed = auc
    pd.DataFrame(controls).to_csv(directory / "circularity_negative_controls.csv", index=False)
    null_auc = [row["heldout_auc"] for row in controls[1:]]
    top_label = set(np.argsort(f_label)[-100:])
    top_source = set(np.argsort(f_source)[-100:])
    # Exact additivity explains model arithmetic, not whether the label is true.
    return {
        "component_shap": {
            "method": "Exact interventional/independent linear SHAP on calibrated primary component logit.",
            "formula": "phi_j = a * beta_j * (x_j - mean_train_j); base = calibrated intercept + a * beta dot mean_train",
            "background": "Mean of training TF-IDF only", "additivity_max_error": exact_error,
            "explains_final_consensus": False,
            "assumption": "Interventional independence ignores correlated unigrams/bigrams; not causal evidence.",
        },
        "source_proxy_crosstab": pd.crosstab(pd.Series(y, name="label"), pd.Series(source, name="Reuters_marker")).to_dict(),
        "top100_ANOVA_label_source_overlap": len(top_label & top_source),
        "ANOVA_F_label_source_spearman": float(spearmanr(f_label, f_source).statistic),
        "negative_controls": controls,
        "permutation_p_value": float((1+sum(value >= observed for value in null_auc))/(1+len(null_auc))),
        "permutation_resolution": 1/(1+len(null_auc)),
        "conclusion": (
            "ANOVA is supervised association with corpus labels; SHAP explains the fitted model. "
            "Neither is independent validation. Large F or nonzero SHAP can persist in negative controls. "
            "Publisher-proxy association can reproduce the target's construction."
        ),
        "feature_removal_rule": "Only preregistered source/boilerplate rules; no removal by test/external performance or SHAP magnitude.",
        "anova_caution": "BH FDR adjustment is provided; correlated terms/articles and source-constructed labels limit inferential interpretation. F statistics and q-values are diagnostic, not evidence of factual validity.",
    }


def local_and_stability(committee, test, background, directory):
    sample = test.sample(min(400, len(test)), random_state=SEED).reset_index(drop=True)
    texts = article_texts(sample)
    original = committee.details(texts)
    rows = []
    for condition in ("original", "uppercase", "publisher_prefix", "social_footer", "typos", "headline_only"):
        modified = []
        for i, text in enumerate(texts):
            if condition == "uppercase":
                value = text.upper()
            elif condition == "publisher_prefix":
                value = "WASHINGTON (Reuters) - " + text
            elif condition == "social_footer":
                value = text + " Featured image Getty Images. Follow us on Twitter and Facebook. Subscribe and share."
            elif condition == "headline_only":
                value = str(sample.iloc[i].title)
            elif condition == "typos":
                rng = np.random.default_rng(SEED+i)
                words = text.split()
                for j, word in enumerate(words):
                    if len(word) > 4 and rng.random() < .08:
                        words[j] = word[:1]+word[2]+word[1]+word[3:]
                value = " ".join(words)
            else:
                value = text
            modified.append(value)
        details = committee.details(modified)
        metric = evaluate_predictions(sample.label, details["probabilities"], committee.policy["threshold"])
        rows.append({
            "condition": condition, "n": len(sample), "macro_f1": metric["f1_macro"],
            "class_flip_fraction": float((details["prediction"] != original["prediction"]).mean()),
            "decision_flip_fraction": float((details["decisions"] != original["decisions"]).mean()),
            "mean_absolute_score_change": float(np.abs(details["probabilities"]-original["probabilities"]).mean()),
            "review_fraction": float((details["decisions"] == "review").mean()),
        })
    pd.DataFrame(rows).to_csv(directory / "robustness.csv", index=False)
    # Local examples include an error and a near-threshold case, not only successes.
    errors = np.flatnonzero(original["prediction"] != sample.label.to_numpy())
    uncertain = int(np.argmin(np.abs(original["probabilities"] - committee.policy["threshold"])))
    selected = list(dict.fromkeys(([int(errors[0])] if len(errors) else []) + [uncertain, 0]))[:3]
    local_rows = []
    shap_rows = []
    names = committee.vectorizer.get_feature_names_out()
    for position in selected:
        text = texts[position]
        phi, base, logits = committee.component_shap([text], background)
        order = np.argsort(np.abs(phi[0]))[-12:][::-1]
        for column in order:
            feature = str(names[column])
            shap_rows.append({"id": sample.iloc[position].id, "feature": feature, "component_SHAP": float(phi[0, column]), "base_logit": base, "component_logit": float(logits[0])})
            # Text deletion probes the final ensemble; it is not a causal truth test.
            altered = re.sub(r"\b"+re.escape(feature)+r"\b", " ", clean_text(text), flags=re.I)
            p = float(committee.predict_proba([altered])[0, 1])
            local_rows.append({
                "id": sample.iloc[position].id, "feature": feature,
                "original_score": float(original["probabilities"][position]),
                "score_after_text_deletion": p, "score_delta": p-float(original["probabilities"][position]),
                "label": int(sample.iloc[position].label),
            })
    pd.DataFrame(local_rows).to_csv(directory / "local_consensus_deletions.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(directory / "local_component_shap.csv", index=False)
    return {"results": rows, "local_ids": [str(sample.iloc[i].id) for i in selected], "limitation": "Controlled text perturbations and deletions are not new factual ground truth or causal explanations."}


def threshold_stability(committee, policy_frame, directory, repetitions=60):
    details = committee.details(article_texts(policy_frame))
    rng = np.random.default_rng(SEED)
    labels = policy_frame.label.to_numpy()
    class_indices = [np.flatnonzero(labels == k) for k in (0, 1)]
    rows = []
    for i in range(repetitions):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in class_indices])
        policy, _ = select_consensus_policy(labels[indices], details["probabilities"][indices], details["member_probabilities"][indices])
        rows.append({"bootstrap": i, **policy})
    pd.DataFrame(rows).to_csv(directory / "policy_bootstrap.csv", index=False)
    return {
        "repetitions": repetitions, "enabled_fraction": float(np.mean([r["enabled"] for r in rows])),
        "threshold_ci95": np.quantile([r["threshold"] for r in rows], [.025, .975]).tolist(),
        "consensus_unanimous_fraction": float(np.mean([r["consensus"] == 1 for r in rows])),
        "note": "Reselection stability on policy-validation only. The frozen policy is NOT replaced by bootstrap results.",
    }


def legal_review(directory):
    report = {
        "review_date": "2026-11-07", "scope": "Local academic prototype in Moldova, assumed from course context.",
        "status": "Public redistribution and deployment clearance NOT established.",
        "datasets": {
            "ISOT": {"finding": "Public academic download and requested citations; no explicit general redistribution license established.", "source": "https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/"},
            "FakeNewsNet": {"finding": "Authors restrict full corpus distribution for publisher copyright and Twitter policies; repository LICENSE endpoint returned 404. Public minimal CSV availability is not a blanket copyright license.", "source": "https://github.com/KaiDMML/FakeNewsNet"},
        },
        "primary_legal_sources": [
            {"url": "https://www.agepi.gov.md/ro/node/7407", "finding": "Official AGEPI reference to Moldova copyright law 230/2022. Research use and redistribution must be assessed separately."},
            {"url": "https://datepersonale.md/wp-content/uploads/2026/03/Legea-nr.1952024.pdf", "finding": "CNPDCP guidance states law 195/2024 is in force from 23 August 2026; purpose limitation, minimization and retention apply where personal data is processed."},
        ],
        "implemented_controls": [
            "Raw corpora remain in ignored local data folders; no publisher crawling or tweet hydration.",
            "No external personal profiles, sensitive attributes or person-level risk scoring.",
            "New persistent stream logs redact article titles and body text by default.",
            "Statistical reports export article IDs and aggregate results instead of raw article text.",
            "Automatic public verdict/release gate disabled when external evidence fails.",
        ],
        "unresolved": [
            "Explicit dataset redistribution rights and intended public deployment jurisdiction.",
            "Names and other personal data may remain in source news; no claim of complete anonymization.",
            "Existing local v1 logs, if any, predate the new minimization rule and require owner retention review.",
            "A rights-holder/institution review is required for redistribution; this is an issue inventory, not legal certification.",
        ],
    }
    write_json(directory / "legal_review.json", report)
    return report


def run_audit(permutations=19, bootstrap_repetitions=60, activate=True, version_tag="v2"):
    existing_manifest = ARTIFACTS_DIR / "versions" / version_tag / "manifest.json"
    if existing_manifest.exists() and json.loads(existing_manifest.read_text(encoding="utf-8")).get("finalized"):
        raise ValueError("This version is frozen. Use audit_cycle.py --reproduce for a separate, non-activated reproduction.")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_hash = file_sha256(PROTOCOL_PATH)
    previous_external = (AUDIT_DIR / "external_manifest.json").exists() or (PROJECT_DIR / "data" / "raw" / "fakenewsnet" / "politifact_real.csv").exists()
    write_json(AUDIT_DIR / "protocol_registration.json", {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash, "protocol": protocol,
        "external_seen_in_previous_run": previous_external,
        "preflight": "Only availability/schema inspected before registration, not external model performance.",
    })
    articles = load_articles()
    print("Screening temporal roles and headline contamination...", flush=True)
    frame, removed, split_details = audit_partitions(articles)
    removed.to_csv(AUDIT_DIR / "internal_contamination_exclusions.csv", index=False)
    write_json(AUDIT_DIR / "partitions.json", split_details)
    frame[["id", "role", "label"]].to_csv(AUDIT_DIR / "partition_ids.csv", index=False)
    before, _, _, _, _ = fit_variant(frame, False, AUDIT_DIR / "before")
    after, matrices, parts, background, after_tuning = fit_variant(frame, True, AUDIT_DIR / "after")
    test = parts["internal_reused_test"]
    before_internal, _ = evaluate_committee(before, test, AUDIT_DIR, "before_internal")
    after_internal, _ = evaluate_committee(after, test, AUDIT_DIR, "after_internal")
    before_headlines, _ = evaluate_committee(before, test, AUDIT_DIR, "before_internal_headlines", True)
    after_headlines, _ = evaluate_committee(after, test, AUDIT_DIR, "after_internal_headlines", True)
    write_json(AUDIT_DIR / "frozen_before_external.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "before_fingerprint": committee_fingerprint(before), "after_fingerprint": committee_fingerprint(after),
        "before_policy": before.policy, "after_policy": after.policy,
        "protocol_sha256": protocol_hash,
        "rule": "After is the preregistered final candidate regardless of external outcome; no external-driven retuning.",
    })
    candidate_dir = ARTIFACTS_DIR / "versions" / version_tag
    candidate_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"before": before, "after": after, "protocol_hash": protocol_hash}, candidate_dir / "frozen_candidates.joblib", compress=3)
    frozen_before, frozen_after = committee_fingerprint(before), committee_fingerprint(after)
    print("Frozen. Downloading independent RAW headline CSVs...", flush=True)
    external, external_manifest = fetch_external(protocol, articles, AUDIT_DIR)
    external_before, _ = evaluate_committee(before, external, AUDIT_DIR, "before_external", True)
    external_after, _ = evaluate_committee(after, external, AUDIT_DIR, "after_external", True)
    external_by_source = {}
    for source, part in external.groupby("external_source"):
        result, _ = evaluate_committee(after, part, AUDIT_DIR, f"after_external_{source}", True)
        external_by_source[source] = result
    if committee_fingerprint(before) != frozen_before or committee_fingerprint(after) != frozen_after:
        raise RuntimeError("External validation mutated a frozen model or policy.")
    print("ANOVA/SHAP circularity controls and robustness...", flush=True)
    circularity = feature_and_circularity_audit(after, matrices, parts, background, AUDIT_DIR, permutations)
    robustness = local_and_stability(after, test, background, AUDIT_DIR)
    boundary_stability = threshold_stability(after, parts["policy"], AUDIT_DIR, bootstrap_repetitions)
    legal = legal_review(AUDIT_DIR)
    old_names = set(before.vectorizer.get_feature_names_out())
    new_names = set(after.vectorizer.get_feature_names_out())
    feature_changes = [
        {"feature": feature, "reason": "preregistered source/formatting token", "present_after": feature in new_names}
        for feature in sorted(old_names)
        if any(word in BOILERPLATE_TOKENS for word in feature.split())
    ]
    pd.DataFrame(feature_changes).to_csv(AUDIT_DIR / "removed_bad_features.csv", index=False)
    metrics = external_after["metrics"]
    selective = external_after["selective_policy"]
    gate = {
        "external_macro_f1": metrics["f1_macro"] >= .70,
        "external_macro_f1_ci_lower": metrics["macro_f1_ci95"][0] >= .65,
        "external_automatic_fake_precision": selective["fake_precision_wilson_lower"] >= .90,
        "external_automatic_real_precision": selective["real_precision_wilson_lower"] >= .90,
        "external_coverage": selective["coverage"] >= .50,
        "redistribution_rights_confirmed": False,
    }
    release = {
        "status": "research_only", "automatic_user_verdicts_enabled": False,
        "technical_gate_passed": all(value for key, value in gate.items() if key != "redistribution_rights_confirmed"),
        "checks": gate,
        "reason": "Public deployment/redistribution not cleared; external quality gates and source confounding must be reviewed.",
    }
    report = {
        "version": "2.0.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash,
        "problem": protocol["purpose"], "kpi": protocol["external_gate"],
        "partitions": split_details, "contamination_exclusions": len(removed),
        "internal_test_status": "Previously inspected v1 test; reused diagnostic after stronger contamination filtering.",
        "before": {"internal_full_text": before_internal, "internal_headlines": before_headlines, "external_raw_headlines": external_before, "policy": before.policy},
        "after": {"internal_full_text": after_internal, "internal_headlines": after_headlines, "external_raw_headlines": external_after, "policy": after.policy},
        "external_by_source": external_by_source, "external_manifest": external_manifest,
        "circularity": circularity, "robustness": robustness, "boundary_stability": boundary_stability,
        "removed_feature_count": len(feature_changes), "legal": legal, "release": release,
        "freeze_verified": True, "external_previously_seen": previous_external,
        "methodological_cycle": [
            "Problem", "KPI", "Features", "Model and metric tuning", "Agreement and confusion matrices",
            "Calibrated consensus boundary", "Global and local XAI", "Robustness",
            "Independent raw-headline validation", "Legal inventory",
            "Preregistered bad-feature removal", "Recalculation on same frozen protocol", "Version fixation",
        ],
        "order_note": "Before/after rules and models were registered/fitted before external access; external outcomes do not select features.",
    }
    write_json(AUDIT_DIR / "audit_report.json", report)
    primary = after.members[0]["estimator"]
    primary_pipeline = Pipeline([("clean", after.cleaner), ("tfidf", after.vectorizer), ("classifier", primary)])
    details = after.details(article_texts(test))
    test_predictions = test[["id", "label", "subject", "published_at", "title"]].reset_index(drop=True)
    test_predictions["probability_fake"] = details["probabilities"]
    test_predictions["prediction"] = details["prediction"]
    comparison = pd.DataFrame([
        {"model": f"{row['family']} {row['parameters']}", "validation_f1_macro": row["tuning_macro_f1"],
         "test_f1_macro": None, "test_accuracy": None, "selected": False}
        for row in after_tuning
    ])
    metadata = {
        "version": "2.0.0", "created_at_utc": report["created_at_utc"],
        "dataset": "ISOT + independent FakeNewsNet headline validation", "language": "English",
        "selected_model": "Frozen calibrated consensus: Logistic / SGD / ComplementNB",
        "test_metrics": after_internal["metrics"], "external_metrics": metrics,
        "train_rows": split_details["train"]["n"],
        "validation_rows": sum(value["n"] for role, value in split_details.items() if role not in ("train", "internal_reused_test")),
        "test_rows": len(test), "corpus_sha256": file_sha256(PROJECT_DIR / "data" / "processed" / "articles.parquet"),
        "selection_rule": "Train only; hyperparameters on tuning; Platt calibrators and policy on disjoint chronological roles.",
        "preprocessing_version": "audit-boilerplate-v2", "release": release, "protocol_sha256": protocol_hash,
        "limitations": ["External data are headlines, not full articles.", "Consensus is correlated and cannot establish factual truth.", "The old ISOT test is reused.", "Redistribution rights not established."],
    }
    bundle = {
        "schema_version": 2, "pipeline": primary_pipeline, "consensus": after,
        "background_mean": background, "comparison": comparison, "metadata": metadata,
        "threshold": after.policy["threshold"], "consensus_policy": after.policy,
        "test_predictions": test_predictions,
        "reference_scores": after.details(article_texts(parts["policy"]))["probabilities"],
        "audit_report": str((AUDIT_DIR / "audit_report.json").resolve()),
    }
    version_model = candidate_dir / "model.joblib"
    save_bundle(bundle, version_model)
    source_files = sorted((PROJECT_DIR / "news15").glob("*.py")) + sorted((PROJECT_DIR / "scripts").glob("*.py")) + [PROJECT_DIR / "app.py", PROJECT_DIR / "requirements.txt", PROTOCOL_PATH]
    source_manifest = {}
    for source in source_files:
        relative = source.relative_to(PROJECT_DIR)
        target = candidate_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_manifest[str(relative)] = file_sha256(source)
    write_json(candidate_dir / "manifest.json", {
        "version": "2.0.0", "model_sha256": file_sha256(version_model),
        "protocol_sha256": protocol_hash, "parameter_fingerprint": frozen_after,
        "external_commit": protocol["external_commit"], "external_manifest_sha256": file_sha256(AUDIT_DIR / "external_manifest.json"),
        "report_sha256": file_sha256(AUDIT_DIR / "audit_report.json"), "source_files": source_manifest,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "streamlit", "joblib")},
    })
    if activate:
        # Activate the exact frozen bytes, so the manifest covers the live model.
        shutil.copy2(version_model, MODEL_PATH)
    create_audit_markdown(report, AUDIT_DIR / "AUDIT.md")
    print(json.dumps({"external_macro_f1": metrics["f1_macro"], "internal_macro_f1": after_internal["metrics"]["f1_macro"], "policy": after.policy, "release": release}, indent=2))
    return report


def create_audit_markdown(report, path):
    before, after = report["before"], report["after"]
    lines = [
        "# Методологический аудит версии 2", "",
        "Задача — помощь ручной проверке английских новостей. Метрики на ISOT не доказывают способность проверять факты.", "",
        "| Проверка | До удаления признаков macro-F1 | После удаления macro-F1 |",
        "|---|---:|---:|",
    ]
    for key, label in (("internal_full_text", "Повторный внутренний test, полные тексты"), ("internal_headlines", "Внутренний test, заголовки"), ("external_raw_headlines", "Внешний FakeNewsNet, raw заголовки")):
        lines.append(f"| {label} | {before[key]['metrics']['f1_macro']:.4f} | {after[key]['metrics']['f1_macro']:.4f} |")
    lines += [
        "", "## Граница решения", "", json.dumps(after["policy"], ensure_ascii=False), "",
        "Участники, их среднее и политика откалиброваны на разных временных подвыборках. Согласие моделей не является независимой проверкой фактов.", "",
        "## Круговая логика", "",
        "ANOVA оценивает связь с корпусной меткой. SHAP объясняет выбранную модель. Использовать их как подтверждение той же метки было бы круговой логикой.",
        f"Пересечение top-100 ANOVA для метки и Reuters-proxy: {report['circularity']['top100_ANOVA_label_source_overlap']}.",
        "Проведены перестановки обучающих меток с повторным отбором ANOVA, проверка аддитивности component-SHAP и проверки согласованных ошибок.", "",
        "## Внешняя проверка", "",
        f"Использовано {report['external_manifest']['evaluated_rows']} заголовков закреплённой ревизии FakeNewsNet. Полные статьи в открытых CSV отсутствуют.",
        "Внешние метки не использовались для выбора моделей, удаления признаков или порогов. Старый test ISOT уже просмотрен в v1.", "",
        "## Правовая проверка и допуск", "",
        "Проверены источники данных, наличие лицензии и официальные источники AGEPI/CNPDCP. Разрешение на публичную перераздачу корпусов не установлено.",
        "Правовой статус и вопросы приведены в legal_review.json. Допуск: research_only, автоматические пользовательские вердикты отключены.", "",
        "## Воспроизведение и фиксация", "",
        "Полный цикл: python scripts/audit_cycle.py. Повторное использование внешнего набора явно регистрируется.",
        "Протокол: protocols/v2.json. Модель, снимок исходников и SHA-256: artifacts/versions/v2/manifest.json.",
        "Параметры модели и политика проверены на неизменность после внешней валидации.",
        "", "## Методологические источники", "",
        "Probability calibration: https://scikit-learn.org/1.8/modules/calibration.html",
        "Threshold tuning and separate data: https://scikit-learn.org/1.8/modules/classification_threshold.html",
        "Independent linear SHAP formula: https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html",
        "External raw CSV and copyright limitations: https://github.com/KaiDMML/FakeNewsNet",
    ]
    Path(path).write_text("\n".join(lines)+"\n", encoding="utf-8")

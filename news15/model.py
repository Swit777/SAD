"""Train on past articles, select on validation, evaluate once on future articles."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from .data import (
    ARTICLE_PATH, PREPROCESSING_VERSION, TextCleaner, article_texts,
    combine_text, dataset_summary, file_sha256, load_articles,
)
from .paths import MODEL_PATH, REPORTS_DIR, ensure_directories

SEED = 42
_STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) - {"no", "not", "nor"})


def evaluate_predictions(y_true, probabilities, threshold=0.5):
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    both_classes = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "f1_macro": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "f1_fake": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
        "average_precision": float(average_precision_score(y, probabilities)) if both_classes else None,
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "confusion_matrix": matrix.tolist(),
        "rows": int(len(y)),
        "threshold": float(threshold),
    }


def select_threshold(y_true, probabilities):
    candidates = np.linspace(0.20, 0.80, 61)
    scored = [
        (f1_score(y_true, np.asarray(probabilities) >= t, average="macro"), -abs(t - 0.5), t)
        for t in candidates
    ]
    return float(max(scored)[2])


def _pipeline(classifier, max_features):
    return Pipeline([
        ("clean", TextCleaner()),
        ("tfidf", TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), min_df=2,
            max_df=0.98, sublinear_tf=True, stop_words=_STOP_WORDS,
            strip_accents="unicode", dtype=np.float64,
        )),
        ("classifier", classifier),
    ])


def save_bundle(bundle, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.joblib")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(path)
    return path


def load_bundle(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Model is not trained. Run: python scripts/train.py")
    bundle = joblib.load(path)
    if bundle.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported model artifact version; retrain the model.")
    return bundle


def train_models(articles=None, max_features=40000, output_path=MODEL_PATH):
    if output_path is not None and Path(output_path) == MODEL_PATH and MODEL_PATH.exists():
        active = load_bundle()
        if active.get("schema_version") == 2:
            raise ValueError("Version 2 uses the audit protocol. Run python scripts/audit_cycle.py; legacy training cannot overwrite it.")
    ensure_directories()
    frame = load_articles() if articles is None else articles.copy()
    required = {"id", "title", "text", "label", "split", "group_id", "published_at"}
    if not required.issubset(frame):
        raise ValueError(f"Missing corpus columns: {required - set(frame.columns)}")
    groups_per_split = frame.groupby("group_id")["split"].nunique()
    if (groups_per_split > 1).any():
        raise ValueError("Leakage: the same content group occurs in multiple splits.")
    parts = {name: frame[frame["split"] == name].copy() for name in ("train", "validation", "test")}
    for name, part in parts.items():
        if part.empty or part["label"].nunique() < 2:
            raise ValueError(f"Both labels are required in the {name} split.")
    if (
        parts["train"]["published_at"].max() >= parts["validation"]["published_at"].min()
        or parts["validation"]["published_at"].max() >= parts["test"]["published_at"].min()
    ):
        raise ValueError("Leakage: publication dates must increase strictly between splits.")
    texts = {name: article_texts(part) for name, part in parts.items()}
    labels = {name: part["label"].to_numpy(int) for name, part in parts.items()}
    candidates = {
        "Logistic regression": LogisticRegression(C=2.0, solver="liblinear", max_iter=1000, random_state=SEED),
        "Logistic regression · balanced": LogisticRegression(
            C=2.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=SEED
        ),
        "Complement Naive Bayes": ComplementNB(alpha=0.5),
        "Majority baseline": DummyClassifier(strategy="prior", random_state=SEED),
    }
    comparison = []
    fitted = {}
    validation_scores = {}
    started_all = perf_counter()
    # All candidates use the identical training-fitted representation. Transform
    # the long articles once rather than rebuilding the vocabulary four times.
    feature_started = perf_counter()
    print("Preparing shared TF-IDF features (training vocabulary only)...", flush=True)
    feature_pipeline = _pipeline(candidates["Logistic regression"], max_features)
    cleaner = feature_pipeline.named_steps["clean"]
    vectorizer = feature_pipeline.named_steps["tfidf"]
    train_cleaned = cleaner.fit_transform(texts["train"], labels["train"])
    matrices = {"train": vectorizer.fit_transform(train_cleaned)}
    for split_name in ("validation", "test"):
        matrices[split_name] = vectorizer.transform(cleaner.transform(texts[split_name]))
    feature_seconds = perf_counter() - feature_started
    for name, classifier in candidates.items():
        print(f"Training {name} on {len(parts['train']):,} articles...", flush=True)
        started = perf_counter()
        classifier.fit(matrices["train"], labels["train"])
        pipeline = Pipeline([("clean", cleaner), ("tfidf", vectorizer), ("classifier", classifier)])
        duration = perf_counter() - started
        validation_probabilities = classifier.predict_proba(matrices["validation"])[:, 1]
        test_probabilities = classifier.predict_proba(matrices["test"])[:, 1]
        validation_metrics = evaluate_predictions(labels["validation"], validation_probabilities)
        test_metrics = evaluate_predictions(labels["test"], test_probabilities)
        comparison.append({
            "model": name, "train_seconds": round(duration, 3),
            "shared_feature_seconds": round(feature_seconds, 3),
            **{f"validation_{k}": v for k, v in validation_metrics.items() if k != "confusion_matrix"},
            **{f"test_{k}": v for k, v in test_metrics.items() if k != "confusion_matrix"},
        })
        if isinstance(classifier, LogisticRegression):
            fitted[name] = pipeline
            validation_scores[name] = validation_probabilities
    comparison_frame = pd.DataFrame(comparison)
    # Deployment is selected only among explainable logistic models using validation.
    selected_name = max(
        fitted,
        key=lambda name: float(comparison_frame.loc[
            comparison_frame["model"] == name, "validation_f1_macro"
        ].iloc[0]),
    )
    pipeline = fitted[selected_name]
    threshold = select_threshold(labels["validation"], validation_scores[selected_name])
    test_started = perf_counter()
    test_probabilities = pipeline.predict_proba(texts["test"])[:, 1]
    latency_ms = (perf_counter() - test_started) * 1000 / len(parts["test"])
    selected_metrics = evaluate_predictions(labels["test"], test_probabilities, threshold)
    comparison_frame["selected"] = comparison_frame["model"] == selected_name
    test_predictions = parts["test"][
        ["id", "label", "subject", "published_at", "title"]
        if "subject" in parts["test"] else ["id", "label", "published_at", "title"]
    ].reset_index(drop=True)
    test_predictions["probability_fake"] = test_probabilities
    test_predictions["prediction"] = (test_probabilities >= threshold).astype(int)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "ISOT", "language": "English", "positive_label": 1,
        "label_names": {"0": "real", "1": "fake"},
        "selected_model": selected_name,
        "selection_rule": "Best validation macro-F1 among logistic regression models at threshold 0.5.",
        "threshold_selection": "Maximum validation macro-F1 on grid [0.20, 0.80]; test labels are not used.",
        "comparison_threshold": 0.5,
        "preprocessing_version": PREPROCESSING_VERSION,
        "seed": SEED,
        "train_rows": len(parts["train"]), "validation_rows": len(parts["validation"]), "test_rows": len(parts["test"]),
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "training_seconds": round(perf_counter() - started_all, 3),
        "shared_feature_seconds": round(feature_seconds, 3),
        "train_seconds_definition": "Classifier fit time; shared text feature preparation is recorded separately.",
        "batch_latency_ms_per_article": round(latency_ms, 4),
        "test_metrics": selected_metrics,
        "dataset_summary": dataset_summary() if articles is None else {},
        "corpus_sha256": file_sha256(ARTICLE_PATH) if articles is None and ARTICLE_PATH.exists() else None,
        "limitations": [
            "Prediction is a classification score, not independent fact verification.",
            "English ISOT articles; Russian and Romanian performance is not validated.",
            "ISOT real/fake labels are confounded with publisher/source.",
            "No political or cultural fairness claim is possible without appropriate annotations.",
            "The model has not been evaluated on current news or previously unseen publishers.",
            "Probability scores are not assumed to be calibrated outside this corpus.",
        ],
    }
    bundle = {
        "schema_version": 1, "pipeline": pipeline, "comparison": comparison_frame,
        "metadata": metadata, "threshold": threshold, "test_predictions": test_predictions,
        "reference_scores": validation_scores[selected_name],
    }
    if output_path is not None:
        output_path = Path(output_path)
        save_bundle(bundle, output_path)
        # Custom output paths (e.g. unit tests) keep their reports isolated.
        report_dir = REPORTS_DIR if output_path == MODEL_PATH else output_path.parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        comparison_frame.to_csv(report_dir / "model_comparison.csv", index=False)
        test_predictions.to_csv(report_dir / "test_predictions.csv", index=False)
        (report_dir / "training_report.json").write_text(
            json.dumps({"metadata": metadata, "threshold": threshold, "comparison": comparison}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return bundle


def predict_article(bundle, title, text, threshold=0.5, review_margin=0.1):
    if not 0 <= threshold <= 1 or not 0 <= review_margin < 0.5:
        raise ValueError("Threshold must be inside [0, 1]; review margin must be in [0, 0.5).")
    combined = combine_text(title, text)
    if not combined.strip():
        raise ValueError("Enter a headline or article text.")
    warnings = []
    if len(combined.split()) < 12:
        warnings.append("Слишком короткий текст: результат может быть нестабильным.")
    cyrillic = len(re.findall(r"[а-яё]", combined, flags=re.I))
    letters = len(re.findall(r"[^\W\d_]", combined, flags=re.UNICODE))
    unsupported_language = bool(letters and cyrillic / letters > 0.05)
    if unsupported_language:
        warnings.append("Текст содержит кириллицу. Модель обучена на английском; этот язык не проверен.")
    if len(combined) > 150000:
        combined = combined[:150000]
        warnings.append("Для обработки использованы первые 150 000 символов.")
    committee = committee_for_input(bundle, text, title)
    member_scores = []
    agreement = None
    review_reason = None
    fusion_explanation = None
    if committee is not None:
        input_limit=getattr(committee.cleaner,"max_chars",None)
        if input_limit and len(combined)>input_limit:
            warnings.append(f"Модель использует начало очищенного текста: до {input_limit} символов.")
        details = committee.details([combined])
        probability = float(details["probabilities"][0])
        predicted = int(details["prediction"][0])
        decision = str(details["decisions"][0])
        threshold = committee.policy["threshold"]
        review_margin = committee.policy["margin"]
        agreement = float(details["agreement"][0])
        member_scores = [
            {"model": member["name"], "probability_fake": float(details["member_probabilities"][0, i])}
            for i, member in enumerate(committee.members)
        ]
        if getattr(committee,'fusion_mode','mean_logit')=='member_logits':
            features=committee.fusion_features(details['member_probabilities'])
            coefficients=committee.fusion.coef_[0]
            contributions=features[0]*coefficients
            for i,entry in enumerate(member_scores):
                entry['fusion_weight']=float(coefficients[i])
                entry['fusion_logit_contribution']=float(contributions[i])
            fusion_explanation={'target':'final consensus logit','intercept':float(committee.fusion.intercept_[0]),
                                'logit':float(committee.fusion.intercept_[0]+contributions.sum())}
        if not bundle["metadata"].get("release", {}).get("automatic_user_verdicts_enabled", False):
            decision = "review"
            review_reason = "Исследовательский режим: внешний контроль не разрешает автоматические вердикты."
    else:
        probability = float(bundle["pipeline"].predict_proba([combined])[0, 1])
        predicted = int(probability >= threshold)
        decision = "review" if abs(probability - threshold) <= review_margin else ("fake" if predicted else "real")
    if unsupported_language:
        decision = "review"
    vectorizer = bundle["pipeline"].named_steps["tfidf"]
    cleaned = bundle["pipeline"].named_steps["clean"].transform([combined])
    if vectorizer.transform(cleaned).nnz == 0:
        decision = "review"
        warnings.append("В тексте нет признаков из словаря модели; требуется ручная проверка.")
    return {
        "probability_fake": probability, "prediction": predicted, "decision": decision,
        "confidence": max(probability, 1 - probability), "warnings": warnings,
        "threshold": float(threshold), "review_margin": float(review_margin),
        "members": member_scores, "agreement": agreement, "review_reason": review_reason,
        "input_mode": input_mode(bundle, title, text),
        "fusion_explanation": fusion_explanation,
    }


def explain_article(bundle, title, text, top_n=12):
    """Exact additive TF-IDF × coefficient contributions to the logistic logit."""
    if bundle.get("consensus") is not None:
        committee = committee_for_input(bundle, text, title)
        phi, base, logits = committee.component_shap([combine_text(title, text)], bundle["background_mean"])
        order = np.argsort(np.abs(phi[0]))[::-1][:top_n]
        values = phi[0, order]
        result = pd.DataFrame({
            "feature": committee.vectorizer.get_feature_names_out()[order],
            "contribution": values, "direction": np.where(values >= 0, "fake", "real"),
        })
        active=committee.vectorize([combine_text(title,text)]).indices
        result['present_in_text']=np.isin(order,active)
        result.attrs.update({
            "intercept": base, "logit": float(logits[0]),
            "all_contributions_sum": float(phi[0].sum()),
            "explanation_target": "calibrated primary component logit, NOT final consensus",
        })
        return result
    pipeline = bundle["pipeline"]
    cleaned = pipeline.named_steps["clean"].transform([combine_text(title, text)])
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    vector = vectorizer.transform(cleaned)
    contributions = vector.toarray()[0] * classifier.coef_[0]
    nonzero = np.flatnonzero(contributions)
    chosen = nonzero[np.argsort(np.abs(contributions[nonzero]))[::-1]][:top_n]
    result = pd.DataFrame({
        "feature": vectorizer.get_feature_names_out()[chosen],
        "contribution": contributions[chosen],
        "direction": np.where(contributions[chosen] >= 0, "fake", "real"),
    })
    result.attrs["intercept"] = float(classifier.intercept_[0])
    result.attrs["logit"] = float(classifier.decision_function(vector)[0])
    result.attrs["all_contributions_sum"] = float(contributions.sum())
    return result


def input_mode(bundle, title, text):
    if bundle.get('short_consensus') is not None:
        if not str(text or '').strip():return 'headline'
        return 'short' if len(combine_text(title,text).split()) < 40 else 'full'
    return 'short' if len(str(text or '').split()) < 40 and bundle.get('headline_consensus') is not None else 'full'


def committee_for_input(bundle, text, title=""):
    """Use separately calibrated short-text outputs without changing legacy models."""
    if bundle.get('short_consensus') is not None:
        mode=input_mode(bundle,title,text)
        return bundle[{'full':'consensus','short':'short_consensus','headline':'headline_consensus'}[mode]]
    if len(str(text or "").split()) < 40 and bundle.get("headline_consensus") is not None:
        return bundle["headline_consensus"]
    return bundle.get("consensus")

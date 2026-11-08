"""Reproducible experiments for the six course requirements.

The experiments use existing article splits.  They never download data, fit on
the held-out split, or turn synthetic feature vectors into supposed articles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .paths import REPORTS_DIR

SEED = 42
SOURCE_PATTERN = re.compile(r"\breuters\b", re.IGNORECASE)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas and numpy values to portable JSON records."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _write_csv(frame: pd.DataFrame, directory: Path, name: str) -> str:
    path = directory / name
    frame.to_csv(path, index=False, encoding="utf-8")
    return str(path.resolve())


def _sample(frame: pd.DataFrame, count: int, seed: int = SEED) -> pd.DataFrame:
    """Deterministic label-stratified sampling, with no duplication."""
    count = min(int(count), len(frame))
    if count < 1:
        raise ValueError("The selected split contains no articles.")
    if count == len(frame):
        return frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    groups = [(label, part) for label, part in frame.groupby("label", sort=True)]
    # Allocate at least one row per observed class when the budget allows it.
    counts = {label: 0 for label, _ in groups}
    if count >= len(groups):
        counts = {label: 1 for label, _ in groups}
    remaining = count - sum(counts.values())
    while remaining:
        label, part = max(
            ((label, part) for label, part in groups if counts[label] < len(part)),
            key=lambda pair: len(pair[1]) / len(frame) * count - counts[pair[0]],
        )
        counts[label] += 1
        remaining -= 1
    parts = [part.sample(n=counts[label], random_state=seed) for label, part in groups if counts[label]]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _metrics(labels: Any, probabilities: Any, threshold: float = 0.5) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    two_classes = len(np.unique(labels)) == 2
    real_count = int((labels == 0).sum())
    fake_count = int((labels == 1).sum())
    false_positives = int(((labels == 0) & (predicted == 1)).sum())
    false_negatives = int(((labels == 1) & (predicted == 0)).sum())
    return {
        "n": int(len(labels)),
        "real_count": real_count,
        "fake_count": fake_count,
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)) if two_classes else None,
        "f1_macro": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)) if two_classes else None,
        "precision_fake": float(precision_score(labels, predicted, zero_division=0)),
        "recall_fake": float(recall_score(labels, predicted, zero_division=0)),
        "f1_fake": float(f1_score(labels, predicted, zero_division=0)),
        "false_positive_rate": false_positives / real_count if real_count else None,
        "false_negative_rate": false_negatives / fake_count if fake_count else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if two_classes else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if two_classes else None,
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _texts(frame: pd.DataFrame) -> list[str]:
    from .data import combine_text

    return [combine_text(str(title), str(body)) for title, body in zip(frame["title"].fillna(""), frame["text"].fillna(""))]


def _cleaner(bundle: dict[str, Any]) -> Any:
    pipeline = bundle["pipeline"]
    if hasattr(pipeline, "named_steps") and "clean" in pipeline.named_steps:
        return clone(pipeline.named_steps["clean"])
    from .data import TextCleaner

    return TextCleaner()


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        max_features=20000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def _classifier(*, weighted: bool = False) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=700,
        class_weight="balanced" if weighted else None,
        random_state=SEED,
    )


def _bias_experiment(
    test: pd.DataFrame, probabilities: np.ndarray, threshold: float, directory: Path
) -> dict[str, Any]:
    evaluated = test.copy()
    evaluated["probability_fake"] = probabilities
    evaluated["word_count"] = evaluated["text"].fillna("").astype(str).str.split().str.len()
    evaluated["length_group"] = pd.cut(
        evaluated["word_count"], [-1, 100, 300, 700, np.inf],
        labels=["0-100 words", "101-300 words", "301-700 words", "701+ words"],
    ).astype(str)
    tables: dict[str, pd.DataFrame] = {}
    for column, name in [("subject", "bias_by_subject.csv"), ("length_group", "bias_by_length.csv")]:
        rows = []
        for value, group in evaluated.groupby(column, dropna=False, observed=True):
            rows.append({"group": str(value), **_metrics(group.label, group.probability_fake, threshold)})
        tables[column] = pd.DataFrame(rows)
        _write_csv(tables[column], directory, name)
    return {
        "by_subject": _records(tables["subject"]),
        "by_length": _records(tables["length_group"]),
        "files": {"subject": str((directory / "bias_by_subject.csv").resolve()), "length": str((directory / "bias_by_length.csv").resolve())},
        "limitation": "These are corpus subgroup quality checks, not evidence of political or demographic fairness. Subject labels are strongly associated with source and target in ISOT; one-class groups have no ROC AUC or balanced accuracy.",
    }


def _perturb(text: str, kind: str, seed: int) -> str:
    rng = np.random.default_rng(seed)
    if kind == "uppercase":
        return text.upper()
    words = text.split()
    if kind == "word_dropout_20pct":
        return " ".join(word for word in words if rng.random() >= 0.20)
    if kind == "character_typos_8pct_words":
        output = []
        for word in words:
            if len(word) >= 4 and rng.random() < 0.08:
                offset = int(rng.integers(0, len(word) - 1))
                word = word[:offset] + word[offset + 1] + word[offset] + word[offset + 2:]
            output.append(word)
        return " ".join(output)
    return text


def _noise_experiment(
    test: pd.DataFrame, texts: list[str], bundle: dict[str, Any], threshold: float, directory: Path
) -> dict[str, Any]:
    rows, details = [], []
    pipeline = bundle.get("consensus", bundle["pipeline"])
    original = pipeline.predict_proba(texts)[:, 1]
    baseline_f1 = _metrics(test.label, original, threshold)["f1_fake"]
    for kind in ["original", "word_dropout_20pct", "uppercase", "character_typos_8pct_words"]:
        perturbed = [_perturb(text, kind, SEED + index) for index, text in enumerate(texts)]
        probabilities = original if kind == "original" else pipeline.predict_proba(perturbed)[:, 1]
        scores = _metrics(test.label, probabilities, threshold)
        rows.append({
            "condition": kind,
            **scores,
            "delta_f1_vs_original": scores["f1_fake"] - baseline_f1,
            "prediction_flip_rate": float(((original >= threshold) != (probabilities >= threshold)).mean()),
            "mean_absolute_probability_change": float(np.abs(original - probabilities).mean()),
        })
        details.extend({"id": str(identifier), "condition": kind, "label": int(label), "probability_fake": float(probability)} for identifier, label, probability in zip(test.id, test.label, probabilities))
    results = pd.DataFrame(rows)
    result_path = _write_csv(results, directory, "noise_results.csv")
    detail_path = _write_csv(pd.DataFrame(details), directory, "noise_predictions.csv")
    return {
        "results": _records(results), "files": {"summary": result_path, "predictions": detail_path},
        "limitation": "Controlled synthetic corruption of the same held-out texts measures robustness to these edits only. Uppercase may have no effect because TF-IDF lowercases text. Corrupted texts are not additional real observations.",
    }


def _ablation_experiment(
    train: pd.DataFrame, test: pd.DataFrame, train_texts: list[str], test_texts: list[str],
    bundle: dict[str, Any], directory: Path,
) -> dict[str, Any]:
    rows, details = [], []
    for condition in ["raw_text", "source_cleaned_text"]:
        steps = []
        if condition == "source_cleaned_text":
            steps.append(("clean", _cleaner(bundle)))
        steps.extend([("tfidf", _vectorizer()), ("classifier", _classifier(weighted=True))])
        model = Pipeline(steps)
        model.fit(train_texts, train.label)
        probabilities = model.predict_proba(test_texts)[:, 1]
        rows.append({"condition": condition, "training_n": int(len(train)), **_metrics(test.label, probabilities)})
        details.extend({"id": str(identifier), "condition": condition, "label": int(label), "probability_fake": float(probability)} for identifier, label, probability in zip(test.id, test.label, probabilities))
    marker_rows = []
    audit_cleaner = _cleaner(bundle)
    clean_train_texts = audit_cleaner.fit_transform(train_texts)
    clean_test_texts = audit_cleaner.transform(test_texts)
    for split_name, frame, values, cleaned_values in [
        ("train", train, train_texts, clean_train_texts),
        ("test", test, test_texts, clean_test_texts),
    ]:
        for label in [0, 1]:
            selected = [text for text, target in zip(values, frame.label) if target == label]
            cleaned_selected = [str(text) for text, target in zip(cleaned_values, frame.label) if target == label]
            marker_rows.append({
                "split": split_name, "label": label, "n": len(selected),
                "reuters_marker_rate": float(np.mean([bool(SOURCE_PATTERN.search(text)) for text in selected])) if selected else None,
                "cleaned_reuters_marker_rate": float(np.mean([bool(SOURCE_PATTERN.search(text)) for text in cleaned_selected])) if cleaned_selected else None,
            })
    results, markers = pd.DataFrame(rows), pd.DataFrame(marker_rows)
    return {
        "results": _records(results), "source_marker_rates": _records(markers),
        "files": {
            "summary": _write_csv(results, directory, "source_marker_ablation.csv"),
            "markers": _write_csv(markers, directory, "source_marker_rates.csv"),
            "predictions": _write_csv(pd.DataFrame(details), directory, "source_marker_predictions.csv"),
        },
        "threshold": 0.5,
        "limitation": "Both models are trained on exactly the same sampled training articles and evaluated on the same held-out articles. Cleaning explicit markers cannot remove all source-style or topic confounding and does not establish factual verification. These sampled ablation models differ from the main full-training model.",
    }


def _numeric_features(
    train_texts: list[str], test_texts: list[str], bundle: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cleaner = _cleaner(bundle)
    # The cleaner, vocabulary, IDF, SVD and scaler all learn from training only.
    clean_train = cleaner.fit_transform(train_texts)
    clean_test = cleaner.transform(test_texts)
    vectorizer = _vectorizer()
    train_tfidf = vectorizer.fit_transform(clean_train)
    test_tfidf = vectorizer.transform(clean_test)
    if train_tfidf.shape[1] < 3 or train_tfidf.shape[0] < 4:
        raise ValueError("Experiments need at least four training articles and three vocabulary terms.")
    dimensions = min(50, train_tfidf.shape[1] - 1, train_tfidf.shape[0] - 1)
    reducer = TruncatedSVD(n_components=dimensions, random_state=SEED)
    train_vectors = reducer.fit_transform(train_tfidf)
    test_vectors = reducer.transform(test_tfidf)
    scaler = StandardScaler()
    train_vectors = scaler.fit_transform(train_vectors)
    test_vectors = scaler.transform(test_vectors)
    return train_vectors, test_vectors, {
        "svd_dimensions": int(dimensions),
        "vocabulary_size": int(train_tfidf.shape[1]),
        "svd_explained_variance_ratio_sum": float(reducer.explained_variance_ratio_.sum()),
        "fit_split": "train only",
    }


def _smote_experiment(
    train: pd.DataFrame, test: pd.DataFrame, train_vectors: np.ndarray, test_vectors: np.ndarray,
    feature_metadata: dict[str, Any], directory: Path,
) -> dict[str, Any]:
    from imblearn.over_sampling import SMOTE

    labels = train.label.to_numpy(dtype=int)
    real = np.flatnonzero(labels == 0)
    fake = np.flatnonzero(labels == 1)
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("The SMOTE experiment needs at least two training examples per class.")
    # 20% fake means fake / (fake + real) = 0.2, not fake / real = 0.2.
    real_n = min(len(real), max(2, len(fake) * 4))
    fake_n = min(len(fake), max(2, real_n // 4))
    rng = np.random.default_rng(SEED)
    selected = np.concatenate([rng.choice(real, real_n, replace=False), rng.choice(fake, fake_n, replace=False)])
    rng.shuffle(selected)
    x_imbalanced, y_imbalanced = train_vectors[selected], labels[selected]
    counts = np.bincount(y_imbalanced, minlength=2)
    minority_n = int(counts.min())
    neighbors = min(5, minority_n - 1)
    smote = SMOTE(random_state=SEED, k_neighbors=neighbors)
    x_smote, y_smote = smote.fit_resample(x_imbalanced, y_imbalanced)
    rows = []
    for name, x_fit, y_fit, weighted in [
        ("imbalanced_unweighted", x_imbalanced, y_imbalanced, False),
        ("imbalanced_class_weight", x_imbalanced, y_imbalanced, True),
        ("imbalanced_smote", x_smote, y_smote, False),
    ]:
        model = _classifier(weighted=weighted)
        model.fit(x_fit, y_fit)
        probabilities = model.predict_proba(test_vectors)[:, 1]
        rows.append({
            "method": name, "fit_n": int(len(y_fit)),
            "fit_real_count": int((y_fit == 0).sum()), "fit_fake_count": int((y_fit == 1).sum()),
            **_metrics(test.label, probabilities),
        })
    results = pd.DataFrame(rows)
    return {
        "results": _records(results), "file": _write_csv(results, directory, "smote_results.csv"),
        "feature_metadata": feature_metadata,
        "controlled_training_real_count": int(counts[0]),
        "controlled_training_fake_count": int(counts[1]),
        "controlled_training_fake_fraction": float(counts[1] / counts.sum()),
        "synthetic_vector_count": int(len(y_smote) - len(y_imbalanced)),
        "smote_k_neighbors": neighbors,
        "threshold": 0.5,
        "limitation": "Original ISOT is almost balanced. This experiment deliberately creates a training imbalance with approximately 20% fake articles. SMOTE operates on numeric training SVD features only; synthetic vectors are not real or generated news articles. Held-out articles retain their original distribution. SMOTE and class weights may improve or worsen results; no improvement is assumed.",
    }


def _boundary_experiment(
    train: pd.DataFrame, test: pd.DataFrame, train_vectors: np.ndarray, test_vectors: np.ndarray,
    directory: Path,
) -> dict[str, Any]:
    # These two columns are training-learned SVD coordinates.  Fitting an actual
    # 2D classifier makes the surface a genuine decision boundary, not an
    # invented projection of the high-dimensional production model.
    x_train, x_test = train_vectors[:, :2], test_vectors[:, :2]
    classifier = _classifier(weighted=True)
    classifier.fit(x_train, train.label)
    probabilities = classifier.predict_proba(x_test)[:, 1]
    all_points = np.vstack([x_train, x_test])
    low, high = np.quantile(all_points, [0.01, 0.99], axis=0)
    padding = np.maximum((high - low) * 0.12, 0.3)
    low, high = low - padding, high + padding
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 75), np.linspace(low[1], high[1], 75))
    surface = pd.DataFrame({"component_1": xx.ravel(), "component_2": yy.ravel()})
    surface["probability_fake"] = classifier.predict_proba(surface.to_numpy())[:, 1]
    point_frames = []
    for split_name, frame, vectors in [("train", train, x_train), ("test", test, x_test)]:
        point_frames.append(pd.DataFrame({
            "id": frame.id.astype(str), "split": split_name, "label": frame.label.astype(int),
            "component_1": vectors[:, 0], "component_2": vectors[:, 1],
            "probability_fake": classifier.predict_proba(vectors)[:, 1],
        }))
    points = pd.concat(point_frames, ignore_index=True)
    return {
        "metrics": _metrics(test.label, probabilities),
        "files": {
            "points": _write_csv(points, directory, "decision_boundary_points.csv"),
            "surface": _write_csv(surface, directory, "decision_boundary_surface.csv"),
        },
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "threshold": 0.5,
        "grid_shape": [75, 75],
        "axis_labels": ["Training SVD component 1 (standardized)", "Training SVD component 2 (standardized)"],
        "limitation": "Educational surrogate: a separate logistic classifier fitted on two training SVD components. This is its real 2D boundary, not the boundary of the full TF-IDF production model. The plotted grid clips extreme coordinates at the 1st/99th percentiles; complete point coordinates remain in CSV. No test labels are used to fit the projection or classifier.",
    }


def run_experiments(
    articles: pd.DataFrame | None = None,
    bundle: dict[str, Any] | None = None,
    output_dir: str | Path = REPORTS_DIR,
    sample_size: int = 1200,
) -> dict[str, Any]:
    """Run bounded, deterministic experiments and persist reviewable reports.

    ``sample_size`` bounds held-out articles (maximum 1000); training is capped
    at 3000.  Caller-supplied data must use the canonical article schema.
    """
    if sample_size < 10:
        raise ValueError("sample_size must be at least 10.")
    if articles is None:
        from .data import load_articles

        articles = load_articles()
    if bundle is None:
        from .model import load_bundle

        bundle = load_bundle()
    if bundle.get("schema_version") == 2:
        # Respect the stronger contamination screen of the audited version.
        allowed_test = set(bundle["test_predictions"]["id"].astype(str))
        articles = articles.loc[(articles["split"] != "test") | articles["id"].astype(str).isin(allowed_test)].copy()
    required = {"id", "title", "text", "label", "subject", "split", "group_id"}
    missing = required.difference(articles.columns)
    if missing:
        raise ValueError(f"Missing canonical article columns: {sorted(missing)}")
    if not set(articles.label.dropna().unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 (real) and 1 (fake).")
    train_full = articles.loc[articles.split == "train"]
    test_full = articles.loc[articles.split == "test"]
    if set(train_full.label.unique()) != {0, 1} or set(test_full.label.unique()) != {0, 1}:
        raise ValueError("Both train and test splits must contain real and fake articles.")
    overlapping = set(train_full.group_id).intersection(test_full.group_id)
    if overlapping:
        raise ValueError("Train/test group leakage detected; prepare disjoint article groups first.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    train = _sample(train_full, min(3000, sample_size * 3))
    test = _sample(test_full, min(1000, sample_size))
    train_texts, test_texts = _texts(train), _texts(test)
    threshold = float(bundle.get("threshold", 0.5))
    probabilities = bundle.get("consensus", bundle["pipeline"]).predict_proba(test_texts)[:, 1]
    train_vectors, test_vectors, feature_metadata = _numeric_features(train_texts, test_texts, bundle)
    results: dict[str, Any] = {
        "metadata": {
            "model_version": bundle.get("metadata", {}).get("version", "1"),
            "seed": SEED, "requested_sample_size": int(sample_size),
            "train_sample_n": int(len(train)), "heldout_sample_n": int(len(test)),
            "heldout_split": "test", "main_model_threshold": threshold,
            "full_article_n": int(len(articles)),
            "original_real_count": int((articles.label == 0).sum()),
            "original_fake_count": int((articles.label == 1).sum()),
            "original_fake_fraction": float((articles.label == 1).mean()),
            "sampling": "Deterministic label-stratified sampling; no oversampling of held-out articles.",
            "model_selection": "Fixed experiment settings; no tuning or fitting on test data. The main model and its threshold must already have been selected using training/validation only.",
            "limitation": "Historical corpus experiments do not establish real-world truth detection, political fairness, or deployment performance. All perturbations and oversampling are explicitly synthetic experiments.",
        },
        "baseline_metrics": _metrics(test.label, probabilities, threshold),
        "bias": _bias_experiment(test, probabilities, threshold, directory),
        "noise": _noise_experiment(test, test_texts, bundle, threshold, directory),
        "source_marker_ablation": _ablation_experiment(train, test, train_texts, test_texts, bundle, directory),
        "smote": _smote_experiment(train, test, train_vectors, test_vectors, feature_metadata, directory),
        "decision_boundary": _boundary_experiment(train, test, train_vectors, test_vectors, directory),
        "streaming_rtap": {
            "implemented_here": False,
            "note": "Streaming replay and adaptive review thresholds belong to the application streaming module. These offline experiments do not claim a live news or fact-checking feed.",
        },
    }
    results["report_file"] = str((directory / "experiments.json").resolve())
    # Round-tripping prevents accidental numpy scalars / timestamps / NaN from
    # escaping into the public API or the report used by the application.
    serializable = json.loads(json.dumps(results, ensure_ascii=False, allow_nan=False))
    report_path = Path(results["report_file"])
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary_report.replace(report_path)
    return serializable

"""Prequential replay of real held-out articles with oracle feedback simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news15.model import load_bundle
from news15.paths import MODEL_PATH, REPORTS_DIR
from news15.streaming import AdaptiveReviewPolicy, drift_summary


def _metrics(frame: pd.DataFrame, variant: str) -> dict:
    labels = frame["label"].to_numpy(int)
    predictions = frame[f"{variant}_prediction"].to_numpy(int)
    automatic = frame[f"{variant}_decision"].ne("review").to_numpy()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "precision_fake": float(precision_score(labels, predictions, zero_division=0)),
        "recall_fake": float(recall_score(labels, predictions, zero_division=0)),
        "review_fraction": float((~automatic).mean()),
        "automatic_fraction": float(automatic.mean()),
        "automatic_accuracy": float(accuracy_score(labels[automatic], predictions[automatic])) if automatic.any() else None,
    }


def evaluate_replay(bundle: dict, limit: int = 600, output_dir: Path = REPORTS_DIR) -> dict:
    """Predict before feedback; use saved probabilities without running inference."""
    if limit < 1:
        raise ValueError("Replay limit must be positive.")
    predictions = bundle.get("test_predictions")
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("The saved model has no test_predictions dataframe; retrain it.")
    required = {"id", "label", "probability_fake", "published_at", "title"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Saved test predictions lack columns: {sorted(required - set(predictions.columns))}")
    predictions = predictions.copy()
    predictions["published_at"] = pd.to_datetime(predictions["published_at"], errors="raise")
    if predictions["published_at"].isna().any() or predictions["id"].isna().any():
        raise ValueError("Every replay article needs a publication date and an identifier.")
    if predictions["id"].duplicated().any():
        raise ValueError("Replay article identifiers must be unique.")
    if not predictions["label"].isin([0, 1]).all():
        raise ValueError("Replay labels must be 0 (real) or 1 (fake).")
    scores = predictions["probability_fake"].to_numpy(float)
    if not (np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()):
        raise ValueError("Saved probabilities must be finite and inside [0, 1].")
    predictions = predictions.sort_values(["published_at", "id"], kind="mergesort").head(limit)
    if predictions.empty:
        raise ValueError("No saved test predictions are available for replay.")

    initial = float(bundle.get("threshold", 0.5))
    parameters = {
        "initial_threshold": initial,
        "review_margin": 0.1,
        "window_size": 100,
        "min_feedback": 12,
        "step": 0.05,
        "min_threshold": min(0.25, initial),
        "max_threshold": max(0.75, initial),
    }
    fixed = AdaptiveReviewPolicy(**parameters)
    adaptive = AdaptiveReviewPolicy(**parameters)
    records = []
    for sequence, row in enumerate(predictions.itertuples(index=False), start=1):
        probability = float(row.probability_fake)
        label = int(row.label)
        before = adaptive.threshold
        # These decisions and labels are fixed BEFORE the current answer is observed.
        record = {
            "sequence": sequence,
            "id": str(row.id),
            "published_at": row.published_at.isoformat(),
            "title": str(row.title),
            "label": label,
            "probability_fake": probability,
            "fixed_threshold": initial,
            "fixed_prediction": int(probability >= initial),
            "fixed_decision": fixed.decide(probability),
            "adaptive_threshold_before": before,
            "adaptive_prediction": int(probability >= before),
            "adaptive_decision": adaptive.decide(probability),
        }
        feedback = adaptive.observe(probability, label, event_id=str(row.id))
        record.update({
            "adaptive_threshold_after": adaptive.threshold,
            "threshold_changed_after_feedback": bool(feedback["changed"]),
            "feedback_count_after": int(feedback["feedback_count"]),
            "feedback_reason": feedback["reason"],
            "feedback_source": "oracle feedback simulation",
        })
        records.append(record)
        if sequence % 100 == 0:
            print(f"Replayed {sequence}/{len(predictions)} archived articles...", flush=True)

    events = pd.DataFrame(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stream_replay.csv"
    json_path = output_dir / "stream_replay.json"
    events.to_csv(csv_path, index=False, encoding="utf-8")
    results = {
        "schema_version": 1,
        "experiment": "prequential_archive_replay",
        "feedback_source": "oracle feedback simulation",
        "events": len(events),
        "requested_limit": limit,
        "real_count": int(events["label"].eq(0).sum()),
        "fake_count": int(events["label"].eq(1).sum()),
        "start_date": events.iloc[0]["published_at"],
        "end_date": events.iloc[-1]["published_at"],
        "initial_threshold": initial,
        "final_threshold": float(adaptive.threshold),
        "threshold_changes": len(adaptive.changes),
        "policy_parameters": parameters,
        "model": {
            key: bundle.get("metadata", {}).get(key)
            for key in ("selected_model", "created_at_utc", "corpus_sha256", "seed", "preprocessing_version")
        },
        "metrics": {variant: _metrics(events, variant) for variant in ("fixed", "adaptive")},
        "drift": drift_summary(bundle.get("reference_scores", []), events["probability_fake"].to_numpy()),
        "threshold_history": adaptive.changes,
        "protocol": {
            "ordering": "Publication date ascending, then article id ascending; first N test articles.",
            "probabilities": "Previously saved test probabilities; no repeated inference or model fitting.",
            "prediction_timing": "Current predictions and review decisions are fixed before observing the current label.",
            "feedback_timing": "The dataset label becomes oracle feedback immediately after each prediction.",
            "adapted_parameter": "Decision threshold only; classifier coefficients remain fixed.",
            "reference": "Saved validation probabilities are used solely for score-distribution monitoring.",
            "metric_scope": "Binary metrics cover every event, including its underlying class prediction when sent for review. automatic_accuracy covers only non-review decisions.",
        },
        "limitations": [
            "Oracle corpus labels simulate immediate perfect feedback; this is not a live user-feedback or fact-checking evaluation.",
            "Threshold adaptation uses earlier test labels sequentially. These are prequential metrics, not unchanged offline hold-out metrics.",
            "The first N chronological test articles may not represent the complete test distribution.",
            "No improvement is guaranteed, and ISOT source-confounding and language limitations remain.",
            "Drift monitors scores, not truth or verified changes in accuracy.",
        ],
        "files": {"csv": str(csv_path.resolve()), "json": str(json_path.resolve())},
    }
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=600, help="First N chronological test articles; default: 600.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    try:
        result = evaluate_replay(load_bundle(args.model_path), args.limit, args.output_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({
        key: result[key] for key in ("events", "initial_threshold", "final_threshold", "threshold_changes", "metrics", "files")
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

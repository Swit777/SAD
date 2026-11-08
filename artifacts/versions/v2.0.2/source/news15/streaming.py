"""Archive replay, score monitoring, and bounded adaptation from confirmed labels."""

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score

from .paths import REPORTS_DIR


class AdaptiveReviewPolicy:
    """Adapt only the decision threshold, using explicit recent labelled feedback.

    This does not retrain the text model or infer truth from unlabelled drift.
    Evaluation metrics from the original test set remain unchanged.
    """

    def __init__(
        self, initial_threshold=0.5, review_margin=0.1, window_size=100,
        min_feedback=12, step=0.05, min_threshold=0.25, max_threshold=0.75,
    ):
        if not 0 < min_threshold <= initial_threshold <= max_threshold < 1:
            raise ValueError("Initial threshold must be inside the configured bounds.")
        if not 0 <= review_margin < 0.5:
            raise ValueError("Review margin must be in [0, 0.5).")
        if window_size < min_feedback or min_feedback < 4 or not 0 < step <= 0.25:
            raise ValueError("Invalid feedback window, minimum count, or adaptation step.")
        self.initial_threshold = float(initial_threshold)
        self.threshold = float(initial_threshold)
        self.review_margin = float(review_margin)
        self.window_size = int(window_size)
        self.min_feedback = int(min_feedback)
        self.step = float(step)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.feedback = deque(maxlen=self.window_size)
        self._seen_ids = set()
        self.total_feedback = 0
        self.changes = []

    def decide(self, probability):
        probability = float(probability)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probability must be finite and inside [0, 1].")
        if abs(probability - self.threshold) <= self.review_margin + 1e-12:
            return "review"
        return "fake" if probability >= self.threshold else "real"

    def observe(self, probability, label, event_id=None):
        self.decide(probability)  # Validate score before changing state.
        if label not in (0, 1):
            raise ValueError("Confirmed label must be 0 (real) or 1 (fake).")
        previous = self.threshold
        result = {
            "changed": False, "previous_threshold": previous, "threshold": previous,
            "feedback_count": self.total_feedback, "reason": "insufficient_feedback",
            "f1_before": None, "f1_after": None,
        }
        event_key = str(event_id) if event_id is not None else None
        if event_key is not None and event_key in self._seen_ids:
            result["reason"] = "duplicate_event"
            return result
        if event_key is not None:
            self._seen_ids.add(event_key)
        self.feedback.append({"probability": float(probability), "label": int(label), "event_id": event_key})
        self.total_feedback += 1
        result["feedback_count"] = self.total_feedback
        if len(self.feedback) < self.min_feedback:
            return result
        labels = np.asarray([item["label"] for item in self.feedback])
        if len(np.unique(labels)) < 2:
            result["reason"] = "both_labels_required"
            return result
        scores = np.asarray([item["probability"] for item in self.feedback])
        # Evaluate candidate moves inside one step; no unsupported jump is possible.
        lower = max(self.min_threshold, previous - self.step)
        upper = min(self.max_threshold, previous + self.step)
        candidates = np.unique(np.r_[np.linspace(lower, upper, 21), previous])
        before = float(f1_score(labels, scores >= previous, average="macro", zero_division=0))
        scored = [
            (float(f1_score(labels, scores >= candidate, average="macro", zero_division=0)),
             -abs(candidate - previous), float(candidate))
            for candidate in candidates
        ]
        after, _, candidate = max(scored)
        result.update({"f1_before": before, "f1_after": after, "reason": "no_improvement"})
        if after > before + 1e-6:
            self.threshold = candidate
            result.update({"changed": True, "threshold": candidate, "reason": "confirmed_feedback_improvement"})
            self.changes.append({
                "feedback_count": self.total_feedback, "previous_threshold": previous,
                "threshold": candidate, "f1_before": before, "f1_after": after,
            })
        return result

    update = observe

    @property
    def stats(self):
        return {
            "threshold": self.threshold, "initial_threshold": self.initial_threshold,
            "feedback_count": self.total_feedback, "window_count": len(self.feedback),
            "threshold_changes": len(self.changes),
        }

    def to_dict(self):
        return {
            "initial_threshold": self.initial_threshold, "threshold": self.threshold,
            "review_margin": self.review_margin, "window_size": self.window_size,
            "min_feedback": self.min_feedback, "step": self.step,
            "min_threshold": self.min_threshold, "max_threshold": self.max_threshold,
            "feedback": list(self.feedback), "seen_ids": sorted(self._seen_ids),
            "total_feedback": self.total_feedback, "changes": self.changes,
        }

    @classmethod
    def from_dict(cls, state):
        parameters = {
            key: state[key] for key in (
                "initial_threshold", "review_margin", "window_size", "min_feedback",
                "step", "min_threshold", "max_threshold",
            )
        }
        policy = cls(**parameters)
        threshold = float(state["threshold"])
        if not policy.min_threshold <= threshold <= policy.max_threshold:
            raise ValueError("Stored threshold is outside the configured bounds.")
        policy.threshold = threshold
        policy.feedback.extend(state.get("feedback", []))
        policy._seen_ids = set(state.get("seen_ids", []))
        policy.total_feedback = int(state.get("total_feedback", len(policy.feedback)))
        policy.changes = list(state.get("changes", []))
        return policy


def drift_summary(reference, current):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    result = {
        "reference_count": len(reference), "current_count": len(current),
        "ks_statistic": None, "p_value": None, "score_shift": None, "warning": False,
        "interpretation": "Нужно не менее 20 наблюдений в каждом наборе.",
    }
    if len(reference) < 20 or len(current) < 20:
        return result
    statistic, p_value = ks_2samp(reference, current)
    shift = float(current.mean() - reference.mean())
    warning = bool(p_value < 0.01 and statistic > 0.15)
    result.update({
        "ks_statistic": float(statistic), "p_value": float(p_value),
        "score_shift": shift, "warning": warning,
        "interpretation": (
            "Распределение оценок изменилось. Это сигнал для проверки; без подтверждённых меток изменение качества неизвестно."
            if warning else
            "Выраженное изменение распределения оценок не обнаружено."
        ),
    })
    return result


def append_event(event, path=None):
    path = Path(path) if path is not None else REPORTS_DIR / "stream_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Persistent telemetry needs scores/IDs, not potentially personal article text.
    minimized = {key: value for key, value in event.items() if key not in ("title", "text", "body", "news_url", "tweet_ids")}
    record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), **minimized}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str, allow_nan=False) + "\n")
    return path

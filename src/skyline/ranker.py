"""Explainable value-aware ranking, optimized for the top of the list."""

import math
from typing import Dict, Iterable, List, Optional

from .features import light_curve_features
from .model import LinearRanker
from .spatial import SphericalGridIndex
from .types import Candidate, CrossMatch, Observation


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _novelty(match: Optional[CrossMatch]) -> float:
    if match is None:
        return 1.0
    if match.catalog.lower() in {"minor_planet_center", "mpc"}:
        return 0.02
    if match.kind.lower() in {"variable_star", "star", "known_variable"}:
        return 0.12
    return 0.45


def score_object(points: List[Observation], index: SphericalGridIndex, model: Optional[LinearRanker] = None) -> Candidate:
    points = sorted(points, key=lambda point: point.jd)
    current = points[-1]
    features = light_curve_features(points)
    match = index.nearest(current)
    age_days = max(0.0, current.jd - current.first_detected_jd)
    novelty = _novelty(match)

    # A transparent baseline. These weights are a starting policy, not a claim
    # of astronomical calibration. JAX can fit a replacement from delayed
    # historical labels while retaining the same value/recency decision layer.
    rise_signal = math.tanh(features["rise_rate"] * 2_000_000)
    decline_signal = math.tanh(-features["decay_rate"] * 2_000_000)
    color_signal = min(1.0, abs(features["color_evolution"]) * 2.5)
    quality = min(1.0, features["observations"] / 5.0) * max(0.0, 1.0 - features["mean_magnitude_error"] * 3)
    model_features = {
        "novelty": novelty,
        "periodicity": features["periodicity"],
        "rise_rate": features["rise_rate"],
        "decay_rate": features["decay_rate"],
        "color_evolution": features["color_evolution"],
        "quality": quality,
        "mean_magnitude_error": features["mean_magnitude_error"],
    }
    if model is None:
        logit = -1.6 + 2.8 * novelty + 1.3 * rise_signal + 0.5 * decline_signal + 0.45 * color_signal + 0.8 * quality - 3.5 * features["periodicity"] - 5.0 * features["mean_magnitude_error"]
        confidence = _sigmoid(logit)
    else:
        confidence = model.predict(model_features)
    # Scientific value decays once a candidate has sat in the queue. Keep this
    # separate from confidence: an old, genuine transient can be low-value.
    freshness = math.exp(-age_days / 5.0)
    value_score = confidence * freshness

    reasons = []
    if match is None:
        reasons.append("no catalogued counterpart within uncertainty-aware radius")
    else:
        reasons.append(f"matched {match.kind} in {match.catalog} at {match.separation_arcsec:.1f} arcsec")
    if rise_signal > 0.2:
        reasons.append("brightening trend is consistent with an early transient")
    if features["periodicity"] > 0.35:
        reasons.append("periodicity penalty applied")
    if age_days <= 1.0:
        reasons.append("fresh detection receives high follow-up value")
    if model is not None:
        reasons.append("confidence supplied by the trained JAX ranking model")
    return Candidate(current.object_id, current.ra_deg, current.dec_deg, age_days, features, match, confidence, value_score, reasons)


def rank_objects(objects: Dict[str, List[Observation]], index: SphericalGridIndex, budget: int = 20, model: Optional[LinearRanker] = None) -> List[Candidate]:
    candidates = [score_object(points, index, model) for points in objects.values()]
    return sorted(candidates, key=lambda candidate: candidate.value_score, reverse=True)[:budget]


def precision_at_k(ranked_ids: Iterable[str], confirmed: Dict[str, str], k: int) -> float:
    top = list(ranked_ids)[:k]
    if not top:
        return 0.0
    # Confirmation label ingestion can become more granular (e.g. classes with
    # different science value); this first metric is deliberately unforgiving.
    return sum(object_id in confirmed for object_id in top) / len(top)

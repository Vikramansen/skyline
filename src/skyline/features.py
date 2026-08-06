"""Features for sparse, irregular and noisy multi-band light curves."""

import math
from typing import Dict, List, Sequence

from .types import Observation


def _weighted_slope(points: Sequence[Observation]) -> float:
    if len(points) < 2:
        return 0.0
    weights = [1.0 / (point.magnitude_error ** 2) for point in points]
    t0 = points[0].jd
    times = [point.jd - t0 for point in points]
    values = [point.flux for point in points]
    weight_sum = sum(weights)
    mean_t = sum(weight * time for weight, time in zip(weights, times)) / weight_sum
    mean_y = sum(weight * value for weight, value in zip(weights, values)) / weight_sum
    denominator = sum(weight * (time - mean_t) ** 2 for weight, time in zip(weights, times))
    if denominator == 0:
        return 0.0
    return sum(weight * (time - mean_t) * (value - mean_y) for weight, time, value in zip(weights, times, values)) / denominator


def lomb_scargle_peak(points: Sequence[Observation], min_period_days: float = 0.2, max_period_days: float = 30.0) -> float:
    """Return a compact, error-weighted periodicity proxy in [0, 1].

    This is intentionally a dependency-free Lomb-Scargle-style projection for
    laptop demos. Replace it with ``astropy.timeseries.LombScargle`` for a
    scientific run; the feature contract remains the same.
    """
    if len(points) < 4:
        return 0.0
    weights = [1.0 / point.magnitude_error ** 2 for point in points]
    total_weight = sum(weights)
    mean = sum(weight * point.flux for weight, point in zip(weights, points)) / total_weight
    variance = sum(weight * (point.flux - mean) ** 2 for weight, point in zip(weights, points))
    if variance <= 0:
        return 0.0
    best = 0.0
    for step in range(64):
        period = min_period_days * (max_period_days / min_period_days) ** (step / 63)
        omega = 2 * math.pi / period
        cosine = [math.cos(omega * point.jd) for point in points]
        sine = [math.sin(omega * point.jd) for point in points]
        cosine_power = sum(weight * value * basis for weight, value, basis in zip(weights, [p.flux - mean for p in points], cosine)) ** 2
        sine_power = sum(weight * value * basis for weight, value, basis in zip(weights, [p.flux - mean for p in points], sine)) ** 2
        normalizer = sum(weight * basis * basis for weight, basis in zip(weights, cosine)) + sum(weight * basis * basis for weight, basis in zip(weights, sine))
        if normalizer:
            best = max(best, (cosine_power + sine_power) / (variance * normalizer))
    return min(best, 1.0)


def light_curve_features(points: List[Observation]) -> Dict[str, float]:
    if not points:
        raise ValueError("Cannot featurize an empty light curve")
    points = sorted(points, key=lambda point: point.jd)
    peak_index = max(range(len(points)), key=lambda index: points[index].flux)
    rise = _weighted_slope(points[:peak_index + 1])
    decay = _weighted_slope(points[peak_index:])
    by_band = {}
    for point in points:
        by_band.setdefault(point.band, []).append(point)
    colors = []
    if len(by_band) >= 2:
        first_band, second_band = sorted(by_band)[:2]
        left, right = by_band[first_band], by_band[second_band]
        # Compare first/last observed flux ratios: robust to different cadence.
        first_color = math.log10(left[0].flux / right[0].flux)
        last_color = math.log10(left[-1].flux / right[-1].flux)
        colors.append(last_color - first_color)
    errors = [point.magnitude_error for point in points]
    return {
        "periodicity": lomb_scargle_peak(points),
        "rise_rate": rise,
        "decay_rate": decay,
        "color_evolution": sum(colors) / len(colors) if colors else 0.0,
        "mean_magnitude_error": sum(errors) / len(errors),
        "observations": float(len(points)),
    }

"""Portable inference contract for models trained with JAX."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


FEATURE_NAMES = ("novelty", "periodicity", "rise_rate", "decay_rate", "color_evolution", "quality", "mean_magnitude_error")
MODEL_VERSION = 1


@dataclass(frozen=True)
class LinearRanker:
    """A standardised linear model exported by the JAX training backend.

    Inference deliberately uses the Python standard library, so the nightly
    service does not need JAX installed or incur compilation latency.
    """
    weights: Sequence[float]
    bias: float
    location: Sequence[float]
    scale: Sequence[float]
    feature_names: Sequence[str] = FEATURE_NAMES

    def __post_init__(self) -> None:
        expected = len(self.feature_names)
        if not all(len(values) == expected for values in (self.weights, self.location, self.scale)):
            raise ValueError("Model weights, location, and scale must match feature_names")
        if any(not math.isfinite(float(value)) for values in (self.weights, self.location, self.scale) for value in values):
            raise ValueError("Model parameters must be finite")
        if not math.isfinite(float(self.bias)):
            raise ValueError("Model bias must be finite")
        if any(float(value) <= 0 for value in self.scale):
            raise ValueError("Model scale values must be positive")

    def predict(self, features: Dict[str, float]) -> float:
        logit = self.bias + sum(
            float(weight) * ((float(features.get(name, 0.0)) - float(location)) / float(scale))
            for name, weight, location, scale in zip(self.feature_names, self.weights, self.location, self.scale)
        )
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_version": MODEL_VERSION,
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "bias": self.bias,
            "location": list(self.location),
            "scale": list(self.scale),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "LinearRanker":
        if payload.get("model_version") != MODEL_VERSION:
            raise ValueError(f"Unsupported Skyline model version: {payload.get('model_version')}")
        if tuple(payload.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError("Model feature contract does not match this Skyline version")
        return cls(
            weights=[float(value) for value in payload["weights"]], bias=float(payload["bias"]),
            location=[float(value) for value in payload["location"]], scale=[float(value) for value in payload["scale"]],
            feature_names=[str(value) for value in payload["feature_names"]],
        )


def load_model(path: str) -> LinearRanker:
    with Path(path).open(encoding="utf-8") as source:
        return LinearRanker.from_dict(json.load(source))

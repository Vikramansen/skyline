"""JAX training backend for Skyline's top-of-list ranking model.

JAX is used where it matters: differentiating a balanced classification loss
plus an all-pairs ranking objective. The trained artifact is then portable and
safe to serve without JAX.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .model import FEATURE_NAMES, LinearRanker


LabeledRow = Tuple[Dict[str, float], int]


def load_labeled_rows(path: str) -> List[LabeledRow]:
    """Read JSONL records of `{ "features": {...}, "label": 0 | 1 }`."""
    rows: List[LabeledRow] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                label = int(payload["label"])
                if label not in (0, 1):
                    raise ValueError("label must be 0 or 1")
                features = {name: float(payload["features"].get(name, 0.0)) for name in FEATURE_NAMES}
                if not all(math.isfinite(value) for value in features.values()):
                    raise ValueError("features must be finite")
                rows.append((features, label))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid training record at {path}:{line_number}") from error
    if len({label for _, label in rows}) != 2:
        raise ValueError("Training requires at least one positive and one negative example")
    return rows


def fit(rows: Sequence[LabeledRow], epochs: int = 500, learning_rate: float = 0.03, l2: float = 0.01, ranking_weight: float = 0.25) -> LinearRanker:
    """Fit regularized weighted-BCE + pairwise logistic ranking loss using JAX."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise RuntimeError("JAX is required for training. Install Skyline with `pip install -e '.[jax]'`.") from error
    if epochs <= 0 or learning_rate <= 0 or l2 < 0 or ranking_weight < 0:
        raise ValueError("epochs and learning_rate must be positive; l2 and ranking_weight cannot be negative")
    if len({label for _, label in rows}) != 2:
        raise ValueError("Training requires at least one positive and one negative example")

    x = jnp.asarray([[features[name] for name in FEATURE_NAMES] for features, _ in rows], dtype=jnp.float32)
    y = jnp.asarray([label for _, label in rows], dtype=jnp.float32)
    location = jnp.median(x, axis=0)
    iqr = jnp.percentile(x, 75, axis=0) - jnp.percentile(x, 25, axis=0)
    # Keep small, physically meaningful quantities (for example flux slopes
    # around 1e-8) scaled by their observed spread. Only a truly constant
    # column receives the neutral scale.
    scale = jnp.where(jnp.abs(iqr) < 1e-12, 1.0, iqr)
    x = (x - location) / scale
    positive_weight = (len(rows) - jnp.sum(y)) / jnp.sum(y)

    def loss(params):
        weights, bias = params
        logits = x @ weights + bias
        # Stable binary cross entropy, with positives up-weighted for the
        # extreme imbalance that makes accuracy a poor objective here.
        bce = jnp.logaddexp(0.0, logits) - y * logits
        weighted_bce = jnp.mean(jnp.where(y > 0, positive_weight, 1.0) * bce)
        # Every positive should outrank every negative. This is a smooth,
        # differentiable proxy for the precision-at-k behavior we report.
        margins = logits[:, None] - logits[None, :]
        positive_negative = y[:, None] * (1.0 - y[None, :])
        pairwise = jnp.sum(jax.nn.softplus(-margins) * positive_negative) / jnp.maximum(1.0, jnp.sum(positive_negative))
        return weighted_bce + ranking_weight * pairwise + l2 * jnp.sum(weights ** 2)

    step = jax.jit(jax.value_and_grad(loss))
    params = (jnp.zeros(len(FEATURE_NAMES), dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
    for _ in range(epochs):
        _, gradients = step(params)
        params = tuple(parameter - learning_rate * gradient for parameter, gradient in zip(params, gradients))
    weights, bias = params
    return LinearRanker(
        weights=[float(value) for value in weights], bias=float(bias),
        location=[float(value) for value in location], scale=[float(value) for value in scale],
    )


def train_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a Skyline ranker with JAX.")
    parser.add_argument("--training-data", required=True, help="Historical JSONL feature rows with delayed labels.")
    parser.add_argument("--output", required=True, help="Output model JSON path.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    args = parser.parse_args()
    model = fit(load_labeled_rows(args.training_data), args.epochs, args.learning_rate, args.l2, args.ranking_weight)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote JAX-trained ranker to {output}")


if __name__ == "__main__":
    train_main()

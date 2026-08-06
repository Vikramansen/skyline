import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .ingest import load_alerts, load_catalog, load_confirmations
from .model import load_model
from .ranker import precision_at_k, rank_objects
from .spatial import SphericalGridIndex
from .types import Candidate


def rank_tonight(alerts_path: str, catalog_path: str, budget: int = 20, model_path: str = None) -> List[Candidate]:
    """Load one alert night, featurize, retrieve known objects, and rank it."""
    model = load_model(model_path) if model_path else None
    return rank_objects(load_alerts(alerts_path), SphericalGridIndex(load_catalog(catalog_path)), budget, model)


def write_run(output: str, candidates: List[Candidate], total_alerts: int, confirmed: Dict[str, str] = None) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objects_considered": total_alerts,
        "ranked": [candidate.as_dict() for candidate in candidates],
    }
    if confirmed is not None:
        payload["precision_at_k"] = precision_at_k([candidate.object_id for candidate in candidates], confirmed, len(candidates))
        payload["confirmed_detections"] = [
            {"object_id": candidate.object_id, "classification": confirmed[candidate.object_id], "rank": rank}
            for rank, candidate in enumerate(candidates, 1) if candidate.object_id in confirmed
        ]
    (destination / "tonight.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (destination / "summary.md").write_text(
        "# Skyline nightly ranking\n\n"
        f"- Objects considered: {total_alerts}\n"
        f"- Follow-up budget: {len(candidates)}\n"
        + (f"- Precision@{len(candidates)}: {payload['precision_at_k']:.1%}\n" if confirmed is not None else "")
        + "\n| Rank | Candidate | Score | Why |\n| ---: | --- | ---: | --- |\n"
        + "\n".join(f"| {rank} | {item.object_id} | {item.value_score:.3f} | {item.reasons[0]} |" for rank, item in enumerate(candidates, 1))
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank one night of transient alerts.")
    parser.add_argument("--alerts", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--confirmed", help="Optional delayed labels for a backtest.")
    parser.add_argument("--output", default="runs/latest")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--model", help="Optional JAX-trained model JSON; omit for the transparent baseline.")
    args = parser.parse_args()
    objects = load_alerts(args.alerts)
    model = load_model(args.model) if args.model else None
    candidates = rank_objects(objects, SphericalGridIndex(load_catalog(args.catalog)), args.budget, model)
    write_run(args.output, candidates, len(objects), load_confirmations(args.confirmed) if args.confirmed else None)
    print(f"Ranked {len(candidates)} objects from {len(objects)} object light curves. Results: {Path(args.output) / 'tonight.json'}")


if __name__ == "__main__":
    main()

"""Small, auditable JSONL ingestion layer.

Production brokers use Avro/Kafka. Keeping this boundary JSONL lets the same
object-level pipeline run against a downloaded night of broker data or a demo.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from .types import CatalogObject, Observation


def _rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from error


def load_alerts(path: str) -> Dict[str, List[Observation]]:
    grouped: Dict[str, List[Observation]] = defaultdict(list)
    for row in _rows(Path(path)):
        observation = Observation(
            object_id=str(row["object_id"]), jd=float(row["jd"]), band=str(row["band"]),
            magnitude=float(row["magnitude"]), magnitude_error=float(row["magnitude_error"]),
            ra_deg=float(row["ra_deg"]), dec_deg=float(row["dec_deg"]),
            first_detected_jd=float(row["first_detected_jd"]),
            positional_error_arcsec=float(row.get("positional_error_arcsec", 1.0)),
        )
        if observation.magnitude_error <= 0:
            raise ValueError(f"{observation.object_id} has a non-positive magnitude error")
        grouped[observation.object_id].append(observation)
    return {object_id: sorted(points, key=lambda point: point.jd) for object_id, points in grouped.items()}


def load_catalog(path: str) -> List[CatalogObject]:
    return [
        CatalogObject(
            catalog=str(row["catalog"]), object_id=str(row["object_id"]), ra_deg=float(row["ra_deg"]),
            dec_deg=float(row["dec_deg"]), positional_error_arcsec=float(row.get("positional_error_arcsec", 1.0)),
            kind=str(row.get("kind", "unknown")),
        )
        for row in _rows(Path(path))
    ]


def load_confirmations(path: str) -> Dict[str, str]:
    return {str(row["object_id"]): str(row["classification"]) for row in _rows(Path(path))}

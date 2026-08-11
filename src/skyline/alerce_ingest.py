"""Download a bounded, reproducible slice of real ZTF alerts from ALeRCE."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FILTER_NAMES = {1: "g", 2: "r", 3: "i"}


def _rows(response: Any) -> List[Dict[str, Any]]:
    """Normalize the client response across its list and paginated forms."""
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        for key in ("items", "results", "data"):
            if isinstance(response.get(key), list):
                return response[key]
    raise ValueError("ALeRCE returned an unexpected response shape")


def _number(row: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def detection_to_alert(oid: str, detection: Dict[str, Any], first_mjd: float) -> Optional[Dict[str, Any]]:
    """Map documented ALeRCE/ZTF detection fields to Skyline's JSONL contract."""
    mjd = _number(detection, "mjd")
    # Corrected values are usable only when their associated uncertainty is
    # sane. ALeRCE may mark unavailable corrections with a sentinel-like large
    # error, in which case raw PSF photometry is the safer observation.
    corrected_magnitude = _number(detection, "magpsf_corr")
    corrected_error = _number(detection, "sigmapsf_corr")
    use_corrected = bool(detection.get("corrected")) and not bool(detection.get("dubious")) and corrected_magnitude is not None and corrected_error is not None and 0 < corrected_error < 1
    magnitude = corrected_magnitude if use_corrected else _number(detection, "magpsf")
    magnitude_error = corrected_error if use_corrected else _number(detection, "sigmapsf")
    ra = _number(detection, "ra", "meanra")
    dec = _number(detection, "dec", "meandec")
    fid = detection.get("fid")
    try:
        band = FILTER_NAMES[int(fid)]
    except (KeyError, TypeError, ValueError):
        return None
    if None in (mjd, magnitude, magnitude_error, ra, dec) or magnitude_error <= 0:
        return None
    return {
        "object_id": oid,
        "jd": mjd + 2_400_000.5,
        "band": band,
        "magnitude": magnitude,
        "magnitude_error": magnitude_error,
        "ra_deg": ra,
        "dec_deg": dec,
        "first_detected_jd": first_mjd + 2_400_000.5,
        "positional_error_arcsec": 1.0,
        "source": "alerce-ztf",
        "measurement_id": detection.get("measurement_id"),
        "drb": detection.get("drb"),
    }


def fetch_alerts(client: Any, first_mjd: float, last_mjd: float, max_objects: int, min_drb: float) -> List[Dict[str, Any]]:
    """Fetch recent objects, then their detection histories, with bounded work."""
    if not first_mjd < last_mjd:
        raise ValueError("first_mjd must be earlier than last_mjd")
    objects = _rows(client.query_objects(
        survey="ztf", format="json", firstmjd=first_mjd, lastmjd=last_mjd,
        page=1, page_size=max_objects, count=False, order_by="lastmjd", order_mode="DESC",
    ))[:max_objects]
    alerts: List[Dict[str, Any]] = []
    for object_row in objects:
        oid = str(object_row["oid"])
        object_first_mjd = _number(object_row, "mjdstarthist", "firstmjd") or first_mjd
        for detection in _rows(client.query_detections(oid, survey="ztf", format="json")):
            drb = _number(detection, "drb")
            # Missing DRB is retained: older alerts do not all have a deep-RB
            # score, whereas explicitly low-quality detections are rejected.
            if drb is not None and drb < min_drb:
                continue
            converted = detection_to_alert(oid, detection, object_first_mjd)
            if converted is not None:
                alerts.append(converted)
    return sorted(alerts, key=lambda row: (row["object_id"], row["jd"]))


def fetch_object_alerts(client: Any, oids: Iterable[str], min_drb: float, max_detections_per_object: int) -> List[Dict[str, Any]]:
    """Fetch documented object IDs directly; useful when broad searches time out."""
    alerts: List[Dict[str, Any]] = []
    for oid in oids:
        detections = _rows(client.query_detections(str(oid), survey="ztf", format="json"))
        mjds = [mjd for detection in detections if (mjd := _number(detection, "mjd")) is not None]
        if not mjds:
            continue
        first_mjd = min(mjds)
        accepted = []
        for detection in detections:
            drb = _number(detection, "drb")
            if drb is not None and drb < min_drb:
                continue
            converted = detection_to_alert(str(oid), detection, first_mjd)
            if converted is not None:
                accepted.append(converted)
        alerts.extend(sorted(accepted, key=lambda row: row["jd"])[-max_detections_per_object:])
    return sorted(alerts, key=lambda row: (row["object_id"], row["jd"]))


def write_snapshot(output: str, alerts: Iterable[Dict[str, Any]], first_mjd: Optional[float], last_mjd: Optional[float], max_objects: int, min_drb: float, query_mode: str = "night") -> Path:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    rows = list(alerts)
    (directory / "alerts.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    # The local ranker accepts an empty catalog. A production run should replace
    # this file with a versioned Gaia/MPC snapshot, not infer catalog matches
    # from the broker's own labels.
    (directory / "catalog.jsonl").write_text("", encoding="utf-8")
    receipt = {
        "source": "ALeRCE ZTF public API",
        "survey": "ztf",
        "query_mode": query_mode,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "first_mjd": first_mjd,
        "last_mjd": last_mjd,
        "max_objects": max_objects,
        "min_drb": min_drb,
        "alert_rows": len(rows),
        "object_count": len({row["object_id"] for row in rows}),
        "catalog_note": "empty local catalog; cross-match results are not scientifically complete",
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a bounded real ZTF sample from ALeRCE.")
    parser.add_argument("--first-mjd", type=float, help="Inclusive ZTF MJD lower bound.")
    parser.add_argument("--last-mjd", type=float, help="Inclusive ZTF MJD upper bound.")
    parser.add_argument("--oids", nargs="+", help="Direct ZTF object IDs; use when a broad nightly query times out.")
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--max-detections-per-object", type=int, default=100)
    parser.add_argument("--min-drb", type=float, default=0.5, help="Reject detections with an explicit lower DRB score.")
    parser.add_argument("--output", default="data/real/latest")
    args = parser.parse_args()
    if not 1 <= args.max_objects <= 500 or args.max_detections_per_object < 4:
        parser.error("--max-objects must be between 1 and 500")
    if args.oids and (args.first_mjd is not None or args.last_mjd is not None):
        parser.error("Use --oids or an MJD range, not both")
    if not args.oids and (args.first_mjd is None or args.last_mjd is None):
        parser.error("Provide --oids or both --first-mjd and --last-mjd")
    try:
        from alerce.core import Alerce
    except ImportError as error:
        raise SystemExit("Install the broker extra first: .venv/bin/python -m pip install -e '.[broker]'") from error
    client = Alerce()
    if args.oids:
        alerts = fetch_object_alerts(client, args.oids, args.min_drb, args.max_detections_per_object)
        snapshot = write_snapshot(args.output, alerts, None, None, len(args.oids), args.min_drb, "object-list")
    else:
        snapshot = write_snapshot(args.output, fetch_alerts(client, args.first_mjd, args.last_mjd, args.max_objects, args.min_drb), args.first_mjd, args.last_mjd, args.max_objects, args.min_drb)
    receipt = json.loads((snapshot / "receipt.json").read_text())
    print(f"Saved {receipt['alert_rows']} real alerts across {receipt['object_count']} objects to {snapshot}")


if __name__ == "__main__":
    main()

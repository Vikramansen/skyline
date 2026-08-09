"""Optional FastAPI surface. Install with `pip install -e '.[api]'`."""

import os
import csv
import io
import json
from pathlib import Path
from typing import Literal

from .operations import ReviewStore, snapshot_id
from .ingest import load_alerts, load_catalog
from .model import load_model
from .ranker import rank_objects
from .spatial import SphericalGridIndex


def _create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, Response
        from pydantic import BaseModel, Field
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install Skyline's API extra: pip install -e '.[api]'") from error
    app = FastAPI(title="Skyline", version="0.1.0")
    store = ReviewStore(os.environ.get("SKYLINE_DB", "runs/skyline.sqlite"))

    class ReviewRequest(BaseModel):
        decision: Literal["follow_up", "watch", "dismiss"]
        note: str = Field(default="", max_length=280)

    def current_ranking(budget: int) -> dict:
        alerts = os.environ.get("SKYLINE_ALERTS", "data/demo_alerts.jsonl")
        catalog = os.environ.get("SKYLINE_CATALOG", "data/demo_catalog.jsonl")
        model_path = os.environ.get("SKYLINE_MODEL")
        model_label = "jax-trained" if model_path else "transparent baseline"
        objects = load_alerts(alerts)
        candidates = rank_objects(objects, SphericalGridIndex(load_catalog(catalog)), budget, load_model(model_path) if model_path else None)
        curves = {
            object_id: [{"jd": point.jd, "magnitude": point.magnitude, "band": point.band} for point in points]
            for object_id, points in objects.items()
        }
        run_id = snapshot_id((alerts, catalog, model_path) if model_path else (alerts, catalog), budget, model_label)
        ranked = [{**candidate.as_dict(), "light_curve": curves[candidate.object_id]} for candidate in candidates]
        store.save_snapshot(run_id, model_label, len(objects), budget, ranked)
        decisions = store.decisions(run_id)
        for item in ranked:
            item["review"] = decisions.get(item["object_id"], {"decision": "unreviewed", "note": "", "updated_at": None})
        receipt_path = Path(alerts).parent / "receipt.json"
        source = {"label": "bundled demo data", "retrieved_at": None}
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                source = {"label": receipt.get("source", "local alert snapshot"), "retrieved_at": receipt.get("retrieved_at")}
            except (OSError, json.JSONDecodeError):
                source = {"label": "local alert snapshot (receipt unreadable)", "retrieved_at": None}
        return {"run_id": run_id, "budget": budget, "objects_considered": len(objects), "model": model_label, "source": source, "ranked": ranked}

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "service": "skyline"}

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(Path(__file__).with_name("web") / "index.html")

    @app.get("/tonight")
    def tonight(budget: int = 20):
        if not 1 <= budget <= 100:
            raise HTTPException(status_code=422, detail="budget must be between 1 and 100")
        return current_ranking(budget)

    @app.post("/v1/candidates/{object_id}/review")
    def review_candidate(object_id: str, run_id: str, review: ReviewRequest):
        try:
            return {"object_id": object_id, **store.record_review(run_id, object_id, review.decision, review.note)}
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str):
        snapshot = store.get_snapshot(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="unknown ranking snapshot")
        decisions = store.decisions(run_id)
        snapshot.pop("ranked_json", None)
        snapshot["ranked"] = [{**item, "review": decisions.get(item["object_id"], {"decision": "unreviewed", "note": "", "updated_at": None})} for item in snapshot["ranked"]]
        return snapshot

    @app.get("/v1/export/observation-plan")
    def export_observation_plan(run_id: str):
        snapshot = store.get_snapshot(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="unknown ranking snapshot")
        decisions = store.decisions(run_id)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["priority", "object_id", "ra_deg", "dec_deg", "value_score", "confidence", "age_days", "review_status", "review_note", "catalog_match", "reasons"])
        writer.writeheader()
        for priority, item in enumerate(snapshot["ranked"], 1):
            review = decisions.get(item["object_id"], {"decision": "unreviewed", "note": ""})
            match = item.get("cross_match")
            writer.writerow({
                "priority": priority, "object_id": item["object_id"], "ra_deg": item["ra_deg"], "dec_deg": item["dec_deg"],
                "value_score": item["value_score"], "confidence": item["confidence"], "age_days": item["age_days"],
                "review_status": review["decision"], "review_note": review["note"],
                "catalog_match": "" if not match else f"{match['catalog']}:{match['object_id']}", "reasons": " | ".join(item["reasons"]),
            })
        return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="skyline-{run_id}-observation-plan.csv"'})

    return app


app = _create_app()

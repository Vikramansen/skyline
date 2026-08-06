"""Optional FastAPI surface. Install with `pip install -e '.[api]'`."""

import os
from pathlib import Path

from .ingest import load_alerts, load_catalog
from .model import load_model
from .ranker import rank_objects
from .spatial import SphericalGridIndex


def _create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install Skyline's API extra: pip install -e '.[api]'") from error
    app = FastAPI(title="Skyline", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "service": "skyline"}

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(Path(__file__).with_name("web") / "index.html")

    @app.get("/tonight")
    def tonight(budget: int = 20):
        alerts = os.environ.get("SKYLINE_ALERTS", "data/demo_alerts.jsonl")
        catalog = os.environ.get("SKYLINE_CATALOG", "data/demo_catalog.jsonl")
        if not 1 <= budget <= 100:
            raise HTTPException(status_code=422, detail="budget must be between 1 and 100")
        model = os.environ.get("SKYLINE_MODEL")
        objects = load_alerts(alerts)
        candidates = rank_objects(objects, SphericalGridIndex(load_catalog(catalog)), budget, load_model(model) if model else None)
        curves = {
            object_id: [{"jd": point.jd, "magnitude": point.magnitude, "band": point.band} for point in points]
            for object_id, points in objects.items()
        }
        return {
            "budget": budget,
            "objects_considered": len(objects),
            "model": "jax-trained" if model else "transparent baseline",
            "ranked": [{**candidate.as_dict(), "light_curve": curves[candidate.object_id]} for candidate in candidates],
        }

    return app


app = _create_app()

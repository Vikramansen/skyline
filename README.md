# Skyline

**Rank the night sky by what is worth pointing a telescope at.**

Skyline is an ML ranking project for the transient-alert problem: a survey emits millions of changing objects, but the global astronomy community has only a few dozen follow-up slots before the sky rotates. Accuracy is a misleading metric here. The output is a small, prioritized queue, optimized for **precision@k** and for the scientific value of catching an event early.

This repository runs entirely on a laptop with a synthetic night included. It is designed so the demo inputs can be replaced by a downloaded ALeRCE/Fink night without changing the ranking core. JAX trains the learned ranker; standard-library inference keeps the nightly service lightweight and predictable.

## Run it

Create an isolated environment once. This avoids modifying Homebrew-managed
Python and includes the JAX, API, and test dependencies:

```bash
cd skyline
make setup
```

Then run the project with the environment activated, or use the `*-venv`
commands shown below:

```bash
source .venv/bin/activate
make tonight
make test
```

## View the dashboard

The dashboard is the visible nightly follow-up queue: it shows ranked targets,
confidence/value, catalog evidence, ranking reasons, and each object's
multi-band light curve.

```bash
make train-venv
make dashboard
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) while the second command is running. Press `Ctrl-C` in that terminal to stop it.

The dashboard is an operations console, not merely a chart: each ranked target can be marked **follow up**, **watch**, or **dismiss** with a short observing note. Decisions persist in `runs/skyline.sqlite` and are tied to an immutable hash of the input files, model, and budget. Use **Export observation plan** to produce a CSV containing priority, coordinates, scores, catalog evidence, and review state.

The first command writes a durable run receipt to `runs/latest/`:

- `tonight.json` — ranked candidates, feature values, catalog evidence, and delayed-label metric
- `summary.md` — a compact follow-up queue suitable for a nightly review

For the optional API:

```bash
make serve-venv
curl http://127.0.0.1:8000/tonight
```

## Train the ranker with JAX

Train against historical feature rows whose labels arrived after the original ranking decision:

```bash
make setup
make train-venv
PYTHONPATH=src .venv/bin/python -m skyline.pipeline \
  --alerts data/demo_alerts.jsonl --catalog data/demo_catalog.jsonl \
  --model artifacts/jax_ranker.json --output runs/jax-demo
```

`make train` uses a tiny synthetic fixture solely to prove the workflow. Real training rows use this JSONL contract:

```json
{"features":{"novelty":1.0,"periodicity":0.08,"rise_rate":1.5e-8,"quality":0.9},"label":1}
```

The JAX objective combines class-balanced binary cross-entropy with a regularized pairwise logistic loss: every confirmed transient is encouraged to score above every non-target. That is a differentiable training signal aligned with the real evaluation question—who reaches the top of the limited follow-up queue? The exported model contains robust median/IQR scaling, weights, and versioned feature names. The pipeline validates it before use and retains the same explicit freshness decay and catalog rationale.

## Download real broker data

Skyline can ingest a bounded snapshot of public ZTF detections through ALeRCE. Choose an actual observation window in Modified Julian Date (MJD), then run:

```bash
make real-data MJD_START=60200 MJD_END=60201
make real-tonight
```

This writes the source rows and a `receipt.json` under `data/real/latest/`; that directory is intentionally ignored by Git. The receipt records exact query bounds, quality filter, row/object counts, and retrieval time so a result can be reproduced. The downloader caps requests at 500 objects and retains only detections with an explicit `drb` score at or above 0.5 (while preserving older records that lack that field).

The dashboard also includes a **Get real data** panel that converts a UTC date to the two MJD values for this command.

The first real-data pass uses an empty local catalog to avoid claiming a cross-match we did not make. It is useful for validating real light-curve ingestion and ranking mechanics, but it is **not** a complete science run until a versioned Gaia/MPC catalog snapshot is supplied.

If an upstream broker times out on a broad historical window, ingest documented ZTF object IDs directly instead:

```bash
make real-object OID=ZTF18abbuksn MIN_DRB=0
PYTHONPATH=src .venv/bin/python -m skyline.pipeline \
  --alerts data/real/object-ZTF18abbuksn/alerts.jsonl \
  --catalog data/real/object-ZTF18abbuksn/catalog.jsonl \
  --model artifacts/jax_ranker.json --output runs/real/object-ZTF18abbuksn
```

`MIN_DRB=0` is appropriate only for this historical demonstration object; retain the default `0.5` quality threshold for a nightly candidate query.

## What happens to one night's alerts

```text
JSONL broker export → object light curves → error-weighted features
  → uncertainty-aware catalog lookup → transient confidence
  → freshness/value adjustment → ranked follow-up budget
  → later confirmations → precision@k receipt
```

The implementation purposefully operates on **objects**, not raw alert rows: an object can produce several alerts through a night, and a telescope should not spend several scarce slots on the same candidate.

## Features and ranking policy

Each light curve is sparse, irregularly sampled, multi-band, and has per-observation uncertainty. Skyline therefore computes:

- a dependency-free, error-weighted Lomb–Scargle-style periodicity proxy; a scientific run should replace this implementation with `astropy.timeseries.LombScargle` while retaining its feature contract;
- error-weighted flux rise and decay slopes;
- change in two-band flux color; and
- quality signals from cadence and photometric uncertainty.

Cross-matching first retrieves nearby catalog candidates with a spherical grid, then performs an exact great-circle comparison. The acceptance radius is `3 × sqrt(alert_error² + catalog_error²)`, clamped to 1–30 arcseconds; there is no one-size-fits-all match radius. The small local grid is a laptop stand-in with the same interface a production HEALPix index would use.

The ranker combines novelty, early brightening, light-curve quality, color change, and a periodicity penalty into a transient confidence. It then applies an explicit exponential freshness decay. That separation is deliberate: an old but genuine supernova can have high confidence and little remaining follow-up value.

## Backtesting without leakage

`data/demo_confirmed.jsonl` represents labels that arrive after the nightly decision. Passing `--confirmed` computes precision@k only after ranking is frozen:

```bash
PYTHONPATH=src python3 -m skyline.pipeline \
  --alerts path/to/night.jsonl --catalog path/to/catalog.jsonl \
  --confirmed path/to/later-confirmations.jsonl --output runs/2026-08-05
```

For real work, retain each night’s immutable ranking receipt and fetch classifications later from an authoritative source such as the Transient Name Server. Never join future classifications into the feature table before a ranking run.

## Data contracts

Alert JSONL rows require `object_id`, `jd`, `band`, `magnitude`, `magnitude_error`, `ra_deg`, `dec_deg`, and `first_detected_jd`. They may include `positional_error_arcsec` (default: 1.0). Catalog rows require `catalog`, `object_id`, `ra_deg`, `dec_deg`, `positional_error_arcsec`, and `kind`.

## Honest next steps

This is a strong ML-system foundation, not a claim of live Rubin performance. Before publishing results, replace the demo with a broker-derived night, use calibrated historical labels to train the JAX ranker, swap the retrieval implementation for HEALPix at catalog scale, benchmark p50/p95/p99 latency, and inspect hundreds of actual light curves. Record all of that in [FINDINGS.md](FINDINGS.md).

## Commit history spread across last 0 days

## Commit history spread across last 9 days
- 4c2d3c8 2026-08-13 21:47:56 -0700 Add tests/test_operations.py
- c31f940 2026-08-12 21:47:56 -0700 Add tests/test_alerce_ingest.py
- 50cccb9 2026-08-11 21:47:56 -0700 Add src/skyline/operations.py
- 5769990 2026-08-10 21:47:56 -0700 Add src/skyline/alerce_ingest.py
- 3440ad0 2026-08-09 21:47:56 -0700 Add src/skyline/web/index.html
- 672e04b 2026-08-08 21:47:56 -0700 Add src/skyline/service.py
- 2571beb 2026-08-07 21:47:56 -0700 Add pyproject.toml
- 2205ca7 2026-08-06 21:47:56 -0700 Add Makefile
- 75490d8 2026-08-05 21:47:56 -0700 Add .gitignore
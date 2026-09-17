# Prefect flows (D)

Four flows that glue the MLOps loop together. Each is a plain Python entrypoint that also works without Prefect (just `python flows/<name>_flow.py`).

## One-time setup

```bash
pip install "prefect>=3.0" mlflow httpx
prefect work-pool create --type process default-process    # one-time
prefect deploy --all --prefect-file flows/prefect.yaml     # registers all 3 deployments
```

Use Prefect Cloud (free tier) or a local server via `infra/prefect/docker-compose.yml`.

## Required env

| Var | Used by | Why |
|---|---|---|
| `MLFLOW_TRACKING_URI` | retrain, promote | reach the tracking server |
| `ASSETS_BUCKET` | promote (GCS mode) | where to upload promoted bundles |
| `REPO_DIR` | promote (PR mode) | path to a clean git checkout — defaults to `.` |
| `CLOUD_RUN_URL` | smoke_test | target of the daily probe |
| `SMOKE_MODEL_NAME` | smoke_test | which asset to invoke (default `drqn_v1`) |
| `SMOKE_MODEL_SOURCE` | smoke_test | `asset` or `run` (default `asset`) |
| `RETRAIN_POLL_SECONDS` | retrain | MLflow poll cadence (default 60s) |
| `RETRAIN_MAX_WAIT_MIN` | retrain | wall-clock cap (default 180min) |
| `RETRAIN_EVAL_METRIC` | retrain | which metric to compare champions on (default `eval_success_rate`) |
| `GCP_PROJECT_ID` / `GCP_REGION` | train_ladder | project + region for `gcloud run jobs` (defaults `garassino-ml` / `europe-west1`) |
| `TRAIN_JOB` | train_ladder | Cloud Run Job name to execute (default `deepmaze-train`) |
| `ARTIFACTS_BUCKET` | train_ladder | bucket holding `result.json` + bundles (default `garassino-ml-artifacts`) |
| `RUNS_PREFIX` / `SERVE_PREFIX` | train_ladder | runs workspace vs serve prefix (defaults `deepmaze-runs/cloudrun/` / `deepmaze/`) |
| `CLOUD_RUN_URL` | train_ladder, smoke_test | backend URL — train_ladder POSTs `/api/resync` after promoting |
| `STAGE_MAX_REPEATS` | train_ladder | per-rung retry cap (default 2) |
| `LADDER_SMOKE` | train_ladder | set → run only the NANO rung on CPU |
| `ASSETS_PREFIX` | promote (GCS mode) | serve prefix prepended to the bundle path (e.g. `deepmaze/`) |

## Flow reference

### `retrain_flow`

**What it does.** Watches MLflow for a *new* finished run that beats the current champion (highest `eval_success_rate`), then calls `promote_flow` to ship it.

**When to trigger.** After kicking off a Colab notebook — the flow waits for the notebook to finish logging. Useful as a "babysit my training" loop.

**Inputs:** `open_pr: bool = True` — passed through to `promote_flow`.

**Side effects:** opens a PR (or pushes to GCS). Otherwise read-only.

**Run locally:**
```bash
export MLFLOW_TRACKING_URI=...
python flows/retrain_flow.py
```

**Caveat:** polls `search_runs` every 60s up to 3h. If MLflow is unreachable for a long stretch, the flow times out — by design.

### `promote_flow`

**What it does.** Given an MLflow run-id, downloads the `assets/<name>/` artifact, validates `config.json` against `REQUIRED_KEYS`, and either:
- **PR mode (default):** copies the bundle into `<REPO_DIR>/assets/`, creates branch `promote/<name>`, pushes, opens a PR with `gh`.
- **GCS mode:** `gsutil rsync` to `gs://${ASSETS_BUCKET}/${ASSETS_PREFIX}<name>/` (the serve prefix — without it the bundle lands at the bucket root and `sync_assets` never sees it).

**When to trigger.** Manually after any Colab run, or automatically by `retrain_flow`.

**Inputs:** `run_id: str`, `open_pr: bool = True`.

**Side effects:** PR mode mutates git history; GCS mode mutates the bucket.

**Run locally:**
```bash
python flows/promote_flow.py <mlflow-run-id>
```

**Caveats:**
- Requires `gh` authenticated and `git` configured (PR mode).
- Requires `gsutil` + ambient GCP creds (GCS mode).
- Doesn't retry on transient MLflow errors — let Prefect handle retries via deployment config.

### `smoke_test_flow`

**What it does.** POSTs to `${CLOUD_RUN_URL}/api/inference` with a known model name and consumes the SSE stream until it sees a `done: true` episode event. Raises if no such event arrives within 60 s.

**When to trigger.** Daily at 09:00 UTC (see `flows/prefect.yaml`). Also useful as a post-deploy probe.

**Inputs:** none (all config via env).

**Side effects:** none — read-only HTTP.

**Run locally:**
```bash
export CLOUD_RUN_URL=https://deepmaze-backend-...run.app
python flows/smoke_test_flow.py
```

**Bootstrap:** fails on first-ever run because no model has been promoted yet. Run `promote_flow` at least once before enabling the schedule.

### `train_ladder_flow`

**What it does.** Climbs a **convergence ladder** (`flows/ladder.py`) on ephemeral Cloud Run Jobs. For each rung it points the job at the rung config (warm-started from the previous rung's best model via `INIT_BUNDLE`), runs it with `gcloud run jobs update … && gcloud run jobs execute … --wait`, reads the `result.json` the job synced to `gs://${ARTIFACTS_BUCKET}/${RUNS_PREFIX}<tag>/`, and consults `has_learned()` to climb / repeat (≤`STAGE_MAX_REPEATS`) / stop. The best learned rung is GCS→GCS copied into the serve prefix and `POST /api/resync` refreshes the viewer.

Unlike `retrain`/`promote`, this flow needs **no queryable MLflow** — persistence is `file://`→GCS, so the learning signal is read straight from `result.json`.

**When to trigger.** When you want a self-contained "train until it learns" loop. `make cloudrun-train` (full ladder) / `make cloudrun-smoke` (`LADDER_SMOKE=1`, NANO rung on CPU, exercises submit→result→promote→resync).

**Inputs:** `run_tag_base: str = "ladder"`.

**Side effects:** executes Cloud Run Jobs (cost), writes GCS (runs + serve prefixes), refreshes the live viewer.

**Run locally:**
```bash
export GCP_PROJECT_ID=garassino-ml GCP_REGION=europe-west1 TRAIN_JOB=deepmaze-train
export ARTIFACTS_BUCKET=garassino-ml-artifacts CLOUD_RUN_URL=https://deepmaze-backend-...run.app
LADDER_SMOKE=1 python flows/train_ladder_flow.py     # CPU smoke first
```

**Prereqs:** `gcloud` authenticated with `run.developer` + `actAs` on the training SA; the job applied via `infra/terraform/cloud_run_job.tf`; `make push-train` done once (public GHCR package). The ladder rungs are starting points — tune them in `flows/ladder.py`.

## Deployment manifest

See [`prefect.yaml`](prefect.yaml) — four deployments, only `smoke-test` is scheduled by default.

## How `flows/` resolves on a Prefect worker

A worker (local-process or Docker-based) needs to import `from flows.promote_flow import promote_flow`. Two ways that works:

1. Run the worker from the repo root — `flows/` is then directly importable.
2. Set `PYTHONPATH=$PWD` in the Prefect deployment's `job_variables`.

If you containerize the worker later, add a `Dockerfile.prefect` that `COPY . /app && WORKDIR /app` and set `PYTHONPATH=/app`.

# Architecture

deepMaze ships as two layers: the **RL playground** (agents, env, viewer) and the **MLOps loop** (Colab → MLflow → Cloud Run → Prefect → CI/CD → traces). This page is the single-screen wiring reference.

## Six subsystems

| Letter | Name | Lives in | Talks to |
|---|---|---|---|
| **A** | Training notebook | `notebooks/train_agent.ipynb` | B (logs), GCS (bundle) |
| **B** | MLflow tracking server | `infra/mlflow/` | Cloud SQL, GCS |
| **C** | Inference container | `Dockerfile.prod` · `docker/` · `infra/cloudrun/` | GCS (assets), F (traces) |
| **D** | Prefect flows | `flows/` · `infra/prefect/` | B (read), C (smoke), GitHub, GCS |
| **D′** | Ephemeral training ladder | `flows/{ladder,train_ladder_flow}.py` · `Dockerfile.train` · `infra/terraform/cloud_run_job.tf` | Cloud Run Job, GCS, C (`/api/resync`) |
| **E** | CI/CD + Slack | `.github/workflows/deploy.yml` | GAR, Cloud Run, Slack |
| **F** | Observability | `web/otel.py` · `docs/observability.md` | Cloud Trace, Cloud Logging |

## Data flow

```
       ┌──────────────────┐
       │ Colab notebook A │
       └────────┬─────────┘
                │ params + metrics + artifacts
                ▼
       ┌──────────────────┐        ┌───────────────────┐
       │  MLflow server B │◀───────│ Prefect retrain D │
       │  (Cloud Run +    │        └─────────┬─────────┘
       │   Cloud SQL +    │                  │ best run-id
       │   GCS artifacts) │                  ▼
       └────────┬─────────┘        ┌───────────────────┐
                │                  │ Prefect promote D │
                │                  └─────────┬─────────┘
                │                            │ PR or gs://
                │                            ▼
                │                  ┌────────────────────┐
                │                  │ assets/<name>/     │
                │                  │ (config + model +  │
                │                  │  viz/replay.webp)  │
                │                  └─────────┬──────────┘
                │                            │
                │                            │ merge → trigger
                │                            ▼
                │                  ┌────────────────────┐
                │                  │ GHA deploy E       │─────▶ Slack
                │                  │ (build → GAR →     │
                │                  │  Cloud Run replace)│
                │                  └─────────┬──────────┘
                │                            │
                │                            ▼
                │                  ┌────────────────────┐
                └─────────────────▶│ Cloud Run backend C│─────▶ Cloud Trace F
                  (inference may   │ (FastAPI + OTEL)   │      Cloud Logging
                   re-query MLflow │                    │
                   in future)      └─────────┬──────────┘
                                             ▲
                                             │ GET /api/inference
                                  ┌──────────┴─────────┐
                                  │ Prefect smoke D    │ (daily 09:00 UTC)
                                  └────────────────────┘
```

## Ephemeral training ladder (D′)

A self-contained "train until it learns" loop, parallel to A→B→D. No MLflow
server in the decision path — persistence is `file://`→GCS, and the climb signal
is the job's `result.json`.

```
 ┌─────────────────────────┐   per rung (warm-start from prev best)
 │ Prefect train_ladder D′ │──┐
 └─────────────────────────┘  │ gcloud run jobs update+execute --wait
            ▲                  ▼
            │         ┌─────────────────────┐
            │         │ Cloud Run Job        │  L4 or CPU (train_gpu_count)
            │         │ deepmaze-train       │  entrypoint.train.sh:
            │         │  train_runpod.py →   │   INIT_BUNDLE(gs://)→local,
            │         │  result.json+bundle  │   train, upload_artifacts.py
            │         └──────────┬───────────┘
            │ read result.json   │ sync
            │                    ▼
            │     gs://…/deepmaze-runs/cloudrun/<tag>/{mlruns,assets,result.json}
            │                    │ has_learned? → GCS→GCS copy of best rung
            │                    ▼
            │     gs://…/deepmaze/<name>/  (serve prefix)
            │                    │ POST /api/resync
            └────────────────────┴──────▶ Cloud Run backend C → /api/models
```

## Integration contracts (load-bearing seams)

| Producer | Artifact | Consumer | Schema lives in |
|---|---|---|---|
| Notebook A | `assets/<name>/{config.json,model.pt,viz/replay.webp}` | Backend C `_load_pretrained` | `web/server.py::list_models` |
| Notebook A | MLflow run (params, metrics, artifacts) | MLflow B, Prefect D | MLflow REST |
| MLflow B | tracking URI + registry | A, D | `MLFLOW_TRACKING_URI` env |
| Prefect D `promote_flow` | promoted bundle in `assets/` (git) OR `gs://${ASSETS_BUCKET}/${ASSETS_PREFIX}<name>/` | GHA E (git path), Backend C startup (GCS path) | `flows/promote_flow.py::REQUIRED_KEYS` |
| Cloud Run Job D′ | `result.json` + bundle in `gs://…/deepmaze-runs/cloudrun/<tag>/` | Prefect ladder D′ (climb decision) | `scripts/train_runpod.py::_summarize` |
| Prefect ladder D′ | promoted best rung in `gs://…/deepmaze/<name>/` + `POST /api/resync` | Backend C `/api/models` | `flows/train_ladder_flow.py` |
| GHA E | new Cloud Run revision | users + smoke flow D | `infra/cloudrun/service.yaml` |
| Backend C | OTEL spans + structured logs | Cloud Trace, Cloud Logging | `web/otel.py` |

## Env var matrix

| Var | Notebook A | MLflow B | Backend C | Prefect D | GHA E | OTEL F |
|---|---|---|---|---|---|---|
| `GCP_PROJECT_ID` |  | (resource path) | read | read | secret | read |
| `GCP_REGION` |  | deploy.sh | (deploy time) |  | var |  |
| `MLFLOW_TRACKING_URI` | required | own URL | optional | required |  |  |
| `BACKEND_STORE_URI` |  | required |  |  |  |  |
| `ARTIFACT_ROOT` |  | required |  |  |  |  |
| `ASSETS_BUCKET` | optional | (artifact bucket sibling) | required | required (GCS mode) | var |  |
| `CORS_ORIGINS` |  |  | required (`*` if unset) |  | var |  |
| `SLACK_WEBHOOK_URL` |  |  |  |  | secret |  |
| `PREFECT_API_KEY` |  |  |  | required (Cloud) |  |  |
| `WIF_PROVIDER` / `WIF_SERVICE_ACCOUNT` |  |  |  |  | secret |  |
| `GAR_REPO` |  | deploy.sh |  |  | secret |  |
| `CLOUD_RUN_SERVICE` |  |  |  |  | secret |  |
| `CLOUD_RUN_SA_EMAIL` |  |  |  |  | secret |  |
| `OTEL_TRACES_EXPORTER` |  |  | activates F | | service.yaml | required |
| `OTEL_SERVICE_NAME` |  |  |  |  | service.yaml | optional |
| `OTEL_SERVICE_VERSION` |  |  |  |  | service.yaml | optional |
| `OTEL_TRACES_SAMPLER` |  |  |  |  | service.yaml | `_sampler_from_env` |
| `OTEL_TRACES_SAMPLER_ARG` |  |  |  |  | service.yaml | `_sampler_from_env` |
| `GOOGLE_APPLICATION_CREDENTIALS` | local only | (workload identity in prod) | local only | local only |  |  |

## Where to look for X

| If you need to… | Look at |
|---|---|
| Add a new agent algorithm | `agents/` + `training/train.py::create_agent` |
| Add a new metric to MLflow | `notebooks/train_agent.ipynb` cell 4 `_on_ep` |
| Add a new env var to Cloud Run | `infra/cloudrun/service.yaml` + `.github/workflows/deploy.yml` env block |
| Add a new Prefect flow | new file in `flows/` + entry in `flows/prefect.yaml` |
| Tune the convergence ladder | `flows/ladder.py` (`LADDER` rungs + `has_learned` gate) |
| Run a self-contained train-until-learn loop | `make cloudrun-smoke` (CPU) → `make cloudrun-train`; `flows/train_ladder_flow.py` |
| Add a new pretrained model | drop into `assets/<name>/` locally OR push via `promote_flow` |
| Change cold-start behavior | `docker/entrypoint.prod.sh` + `infra/cloudrun/service.yaml` startupProbe |
| Change trace sampling | `OTEL_TRACES_SAMPLER_ARG` in `infra/cloudrun/service.yaml` |
| Wire a new external trigger to deploy | `.github/workflows/deploy.yml::on:` |
| Tear down the GCP stack | `docs/deployment-guide.md` → Tear-down section |

## Design spec

For the rationale and OOS decisions: [`superpowers/specs/2026-06-03-deepmaze-mlops-design.md`](superpowers/specs/2026-06-03-deepmaze-mlops-design.md).

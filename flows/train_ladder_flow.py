"""Prefect flow: climb the convergence ladder on ephemeral Cloud Run Jobs.

For each rung: point the Cloud Run Job's env at the rung config (warm-started
from the previous rung's best model in GCS), execute the job to completion, read
the result.json the job synced to GCS, and decide climb / repeat / stop. The
best learned rung is promoted to the serve prefix and the live viewer is
refreshed.

This flow deliberately does NOT use the queryable-MLflow path that retrain_flow
/ promote_flow assume — persistence here is file://→GCS with no server, so the
learning signal is read straight from the job's result.json in the bucket.

Mechanism note: env is applied with `gcloud run jobs update --update-env-vars`
then `gcloud run jobs execute --wait` (two calls) rather than per-execution
overrides, so it works regardless of the installed gcloud's override support.

Env:
    GCP_PROJECT_ID=garassino-ml   GCP_REGION=europe-west1   TRAIN_JOB=deepmaze-train
    ARTIFACTS_BUCKET=garassino-ml-artifacts
    RUNS_PREFIX=deepmaze-runs/cloudrun/   SERVE_PREFIX=deepmaze/
    CLOUD_RUN_URL=https://deepmaze-backend-...run.app   (optional — /api/resync)
    STAGE_MAX_REPEATS=2
    LADDER_SMOKE=                 (set → run only the NANO rung, on CPU)
"""

from __future__ import annotations

import json
import os
import subprocess

import httpx
from prefect import flow, get_run_logger, task

from flows.ladder import LADDER, NANO_RUNG, has_learned

PROJECT = os.environ.get("GCP_PROJECT_ID", "garassino-ml")
REGION = os.environ.get("GCP_REGION", "europe-west1")
JOB = os.environ.get("TRAIN_JOB", "deepmaze-train")
BUCKET = os.environ.get("ARTIFACTS_BUCKET", "garassino-ml-artifacts")
RUNS_PREFIX = os.environ.get("RUNS_PREFIX", "deepmaze-runs/cloudrun/")
SERVE_PREFIX = os.environ.get("SERVE_PREFIX", "deepmaze/")
MAX_REPEATS = int(os.environ.get("STAGE_MAX_REPEATS", "2"))


def _rung_env(rung, run_tag: str, init_bundle: str | None) -> dict:
    """The env the Cloud Run Job runs with for one rung."""
    env = {
        "AGENTS_TO_RUN": rung.agent,
        "RUN_TAG": run_tag,
        "CURRICULUM": rung.curriculum,
        "PARTIAL": str(rung.partial),
        "COLLECT_ALL": "true" if rung.collect_all else "false",
        "OUTPUT_BASE": "/tmp/dm",
        "ARTIFACTS_BUCKET": BUCKET,
        "ARTIFACTS_PREFIX": f"{RUNS_PREFIX}{run_tag}/",
    }
    if init_bundle:
        env["INIT_BUNDLE"] = init_bundle
    env.update(rung.extra_env)
    return env


@task
def run_rung(rung, run_tag: str, init_bundle: str | None) -> str:
    log = get_run_logger()
    env = _rung_env(rung, run_tag, init_bundle)
    kv = ",".join(f"{k}={v}" for k, v in env.items())
    subprocess.check_call(["gcloud", "run", "jobs", "update", JOB,
                           "--region", REGION, "--project", PROJECT,
                           "--update-env-vars", kv])
    log.info(f"executing {JOB} for {rung.name} (tag {run_tag})")
    subprocess.check_call(["gcloud", "run", "jobs", "execute", JOB,
                           "--region", REGION, "--project", PROJECT, "--wait"])
    return f"{RUNS_PREFIX}{run_tag}/"


@task
def fetch_result(run_prefix: str) -> dict | None:
    from google.cloud import storage
    blob = storage.Client().bucket(BUCKET).blob(f"{run_prefix}result.json")
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


@task
def promote(run_prefix: str, bundle_name: str) -> str:
    """GCS→GCS copy of the rung bundle into the serve prefix the viewer reads."""
    from google.cloud import storage
    log = get_run_logger()
    bucket = storage.Client().bucket(BUCKET)
    src_prefix = f"{run_prefix}assets/{bundle_name}/"
    dst_prefix = f"{SERVE_PREFIX}{bundle_name}/"
    n = 0
    for blob in bucket.list_blobs(prefix=src_prefix):
        rel = blob.name[len(src_prefix):]
        if not rel:
            continue
        bucket.copy_blob(blob, bucket, f"{dst_prefix}{rel}")
        n += 1
    log.info(f"promoted {n} files → gs://{BUCKET}/{dst_prefix}")
    return f"gs://{BUCKET}/{dst_prefix}"


@task
def refresh_viewer() -> None:
    url = os.environ.get("CLOUD_RUN_URL")
    if not url:
        return
    try:
        r = httpx.post(f"{url.rstrip('/')}/api/resync", timeout=180)
        get_run_logger().info(f"viewer resync: {r.status_code} {r.text[:200]}")
    except Exception as e:  # noqa: BLE001 — refresh is best-effort
        get_run_logger().warning(f"viewer resync failed: {e}")


@flow(name="train_ladder_flow")
def train_ladder_flow(run_tag_base: str = "ladder") -> dict:
    log = get_run_logger()
    rungs = [NANO_RUNG] if os.environ.get("LADDER_SMOKE") else LADDER
    init_bundle: str | None = None
    best: dict | None = None

    for rung in rungs:
        learned = False
        for attempt in range(MAX_REPEATS):
            run_tag = f"{run_tag_base}-{rung.name}" + (f"-r{attempt}" if attempt else "")
            run_prefix = run_rung(rung, run_tag, init_bundle)
            result = fetch_result(run_prefix)
            agent_res = (result or {}).get("agents", {}).get(rung.agent)
            if has_learned(agent_res, rung.learn_gate):
                learned = True
                bundle = agent_res["bundle_name"]
                best = {"run_prefix": run_prefix, "bundle": bundle, "result": agent_res}
                model_file = "model.best.pt" if agent_res.get("best_model_path") else "model.pt"
                init_bundle = f"gs://{BUCKET}/{run_prefix}assets/{bundle}/{model_file}"
                log.info(f"{rung.name} LEARNED "
                         f"(succ={agent_res['eval_success_rate']:.1%}, "
                         f"revisit={agent_res['eval_revisit_rate']:.1%})")
                break
            log.info(f"{rung.name} below gate (attempt {attempt + 1}/{MAX_REPEATS})")
        if not learned:
            log.warning(f"{rung.name} never cleared its gate — stopping ladder")
            break

    if best:
        gcs = promote(best["run_prefix"], best["bundle"])
        refresh_viewer()
        return {"promoted": gcs, "bundle": best["bundle"], "result": best["result"]}
    log.error("no rung learned — nothing promoted")
    return {"promoted": None}


if __name__ == "__main__":
    train_ladder_flow()

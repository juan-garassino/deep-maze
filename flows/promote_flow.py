"""Prefect flow: promote an MLflow run to assets/<name>/ and trigger redeploy.

Steps:
1. Download the run's `assets/<run_name>/` artifact dir from MLflow.
2. Validate config.json against the schema the backend expects.
3. Either:
     a) push to gs://${ASSETS_BUCKET}/<name>/  (backend hot-syncs on restart), or
     b) commit to git + open PR (default — keeps assets/ versioned).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow
from prefect import flow, get_run_logger, task

from flows.bundle_schema import validate_bundle as _validate_bundle


@task
def download_bundle(run_id: str) -> Path:
    """Download the ``assets/<run_name>`` artifact of MLflow run `run_id` to a temp dir and return its path."""
    log = get_run_logger()
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    name = run.data.params.get("run_name") or run.data.tags.get("mlflow.runName") or run_id
    tmp_root = Path(tempfile.mkdtemp(prefix="deepmaze-bundle-"))
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"assets/{name}", dst_path=str(tmp_root),
    )
    bundle = Path(local)
    if bundle.name != name:
        bundle = bundle / name
    log.info(f"bundle staged at {bundle}")
    return bundle


@task
def validate_bundle(bundle: Path) -> None:
    """Prefect task wrapper around `bundle_schema.validate_bundle`."""
    _validate_bundle(bundle)


@task
def push_to_gcs(bundle: Path, bucket: str) -> str:
    """Rsync `bundle` to ``gs://<bucket>/<bundle name>/`` with gsutil and return that URI."""
    log = get_run_logger()
    name = bundle.name
    subprocess.check_call(["gsutil", "-m", "rsync", "-r",
                           str(bundle), f"gs://{bucket}/{name}/"])
    log.info(f"pushed to gs://{bucket}/{name}/")
    return f"gs://{bucket}/{name}/"


@task
def commit_and_pr(bundle: Path, repo_dir: Path) -> str:
    """Copy `bundle` into ``<repo_dir>/assets/``, commit it on a ``promote/<name>`` branch, push, open a PR with the gh CLI and return its URL."""
    log = get_run_logger()
    name = bundle.name
    target = repo_dir / "assets" / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(bundle, target)
    branch = f"promote/{name}"
    subprocess.check_call(["git", "-C", str(repo_dir), "checkout", "-B", branch])
    subprocess.check_call(["git", "-C", str(repo_dir), "add", str(target)])
    subprocess.check_call(["git", "-C", str(repo_dir), "commit", "-m",
                           f"promote(assets): {name}"])
    subprocess.check_call(["git", "-C", str(repo_dir), "push", "-u", "origin", branch])
    url = subprocess.check_output(
        ["gh", "pr", "create", "--fill", "--head", branch, "--base", "main"],
        cwd=str(repo_dir), text=True,
    ).strip()
    log.info(f"PR opened: {url}")
    return url


@flow(name="promote_flow")
def promote_flow(run_id: str, open_pr: bool = True) -> str:
    """Promote MLflow run `run_id`: download and validate its bundle, then open a PR (default) or push to ``$ASSETS_BUCKET``. Returns the PR URL or GCS URI."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    bundle = download_bundle(run_id)
    validate_bundle(bundle)

    if not open_pr:
        bucket = os.environ["ASSETS_BUCKET"]
        return push_to_gcs(bundle, bucket)

    repo_dir = Path(os.environ.get("REPO_DIR", "."))
    return commit_and_pr(bundle, repo_dir)


if __name__ == "__main__":
    import sys
    promote_flow(run_id=sys.argv[1])

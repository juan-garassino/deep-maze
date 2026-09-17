"""Upload OUTPUT_BASE artifacts to GCS for the ephemeral Cloud Run training job.

Counterpart to docker/sync_assets.py (which DOWNLOADS gs://→local at serve
startup). Walks ${OUTPUT_BASE} and uploads each file to
gs://${ARTIFACTS_BUCKET}/${ARTIFACTS_PREFIX}<relpath>, using google-cloud-storage
(no gsutil / gcloud CLI in the slim training image). Also exposes download_blob()
so the entrypoint can resolve an INIT_BUNDLE gs:// URI to a local path before
training (cross-rung warm-start).

Run modes:
    python docker/upload_artifacts.py                      # upload OUTPUT_BASE → GCS
    python docker/upload_artifacts.py --download GS DEST   # resolve one gs:// blob
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _client():
    from google.cloud import storage
    return storage.Client()


def upload(src_root, bucket_name: str, prefix: str = "") -> int:
    """Upload every file under src_root to gs://bucket/prefix<relpath>."""
    src_root = Path(src_root)
    bucket = _client().bucket(bucket_name)
    n = 0
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src_root).as_posix()
        bucket.blob(f"{prefix}{rel}").upload_from_filename(str(path))
        n += 1
    return n


def download_blob(gs_uri: str, dest) -> str:
    """Download a single gs://bucket/object to a local path."""
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"not a gs:// URI: {gs_uri}")
    bucket_name, _, blob_name = gs_uri[len("gs://"):].partition("/")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _client().bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))
    return str(dest)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "--download":
        try:
            download_blob(argv[1], argv[2])
        except ImportError:
            print("[upload_artifacts] google-cloud-storage not installed; skip",
                  file=sys.stderr)
            return 1
        print(f"[upload_artifacts] downloaded {argv[1]} → {argv[2]}")
        return 0

    bucket_name = os.environ.get("ARTIFACTS_BUCKET")
    if not bucket_name:
        print("[upload_artifacts] ARTIFACTS_BUCKET unset; skip", file=sys.stderr)
        return 0
    src_root = os.environ.get("ARTIFACTS_SRC") or os.environ.get("OUTPUT_BASE")
    if not src_root or not Path(src_root).is_dir():
        print(f"[upload_artifacts] no source dir ({src_root}); skip", file=sys.stderr)
        return 0
    prefix = os.environ.get("ARTIFACTS_PREFIX", "")
    try:
        n = upload(src_root, bucket_name, prefix)
    except ImportError:
        print("[upload_artifacts] google-cloud-storage not installed; skip",
              file=sys.stderr)
        return 0
    print(f"[upload_artifacts] {n} files {src_root} → gs://{bucket_name}/{prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

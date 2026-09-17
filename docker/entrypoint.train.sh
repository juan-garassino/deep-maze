#!/usr/bin/env bash
# Ephemeral Cloud Run training-job entrypoint.
#   1. resolve INIT_BUNDLE (gs:// → local) for cross-rung warm-start
#   2. run the trainer (writes mlruns/, assets/, result.json under OUTPUT_BASE)
#   3. upload everything under OUTPUT_BASE to GCS — ALWAYS, even on crash
#
# The Prefect ladder (flows/train_ladder_flow.py) sets the per-rung env via
# `gcloud run jobs update --update-env-vars` before each execution.
set -uo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/tmp/dm}"
export OUTPUT_BASE
mkdir -p "$OUTPUT_BASE"

echo "=================================================================="
echo "  deepMaze Cloud Run training job"
echo "  OUTPUT_BASE      = $OUTPUT_BASE"
echo "  ARTIFACTS_PREFIX = ${ARTIFACTS_PREFIX:-<unset>}"
echo "  INIT_BUNDLE      = ${INIT_BUNDLE:-<none>}"
python -c 'import torch; print("  cuda available  =", torch.cuda.is_available())' || true
echo "=================================================================="

# 1. Cross-rung warm-start: pull the gs:// model to a local path the trainer reads.
if [[ "${INIT_BUNDLE:-}" == gs://* ]]; then
  if python docker/upload_artifacts.py --download "$INIT_BUNDLE" /tmp/init/model.pt; then
    export INIT_BUNDLE=/tmp/init/model.pt
  else
    echo "WARN: INIT_BUNDLE download failed — training from scratch"
    unset INIT_BUNDLE
  fi
fi

# 2. Train (never abort the upload below).
set +e
python scripts/train_runpod.py
TRAIN_RC=$?
set -e

# 3. Persist — always, even on partial failure (crash-safe artifacts).
python docker/upload_artifacts.py || echo "WARN: artifact upload failed"

echo "training rc=$TRAIN_RC"
exit "$TRAIN_RC"

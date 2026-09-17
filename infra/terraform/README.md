# `infra/terraform/` — show-and-destroy IaC

Minimal Terraform for the deepMaze inference backend on `garassino-ml` / `europe-west1`. Pairs with workspace root `CLAUDE.md` § "GCP architecture".

| File | Resources |
|---|---|
| `main.tf` | Provider + GCS backend at `gs://garassino-op-tf-state/deepmaze/` |
| `variables.tf` | project_id, region, image_tag, ghcr_owner, ghcr_remote_repo, image_name, service_name, assets_bucket, assets_prefix, cors_origins, wif_pool_id, github_repo, **train_job_name, train_image_name, train_gpu_count, train_timeout** |
| `storage.tf` | Shared `garassino-ml-artifacts` bucket — **imported, not created**, `prevent_destroy=true` |
| `iam.tf` | Backend SA + bucket reader (prefix-scoped, incl. list via objectListPrefix) + Trace agent + run.admin/actAs for the CI deploy + WIF impersonation binding |
| `cloud_run.tf` | Cloud Run v2 service (image via AR remote repo proxying GHCR) + public invoker IAM |
| `cloud_run_job.tf` | Ephemeral training **Job** + training SA (objectAdmin scoped to the `deepmaze` stem — runs + serve prefixes) + AR reader + WIF impersonation. **GPU gated on `train_gpu_count`** (0 = CPU today; 1 = L4 once quota lands). Per-run env + image are `ignore_changes` (the Prefect ladder owns env, `make push-train` owns the image). |
| `outputs.tf` | `cloud_run_url`, `sa_email`, `bucket_name`, `assets_prefix`, `train_job_name`, `train_sa_email` |

## One-time setup

1. **TF state bucket** (lives in `garassino-op`, not here):
   ```bash
   gcloud storage buckets create gs://garassino-op-tf-state \
     --project=garassino-op --location=europe-west1 \
     --uniform-bucket-level-access
   ```
2. **WIF pool** (also lives in `garassino-op`; only needed once for the whole workspace — see `garassino-op` IaC).
3. **Shared assets bucket** — if `gs://garassino-ml-artifacts` doesn't exist yet:
   ```bash
   gcloud storage buckets create gs://garassino-ml-artifacts \
     --project=garassino-ml --location=europe-west1 \
     --uniform-bucket-level-access
   ```
4. **Artifact Registry remote repo proxying GHCR** — Cloud Run cannot pull
   `ghcr.io` directly; the image keeps living on GHCR (workspace policy) and
   Cloud Run pulls it through this proxy:
   ```bash
   gcloud artifacts repositories create ghcr-remote \
     --repository-format=docker --mode=remote-repository \
     --remote-docker-repo=https://ghcr.io \
     --location=europe-west1 --project=garassino-ml
   ```
   The GHCR package must be **public** (no upstream credentials configured).
   The same `ghcr-remote` repo serves **both** images: `deepmaze-backend`
   (inference service) and `deepmaze-train-job` (ephemeral training job).

## Ephemeral training job (Cloud Run Job)

`cloud_run_job.tf` adds the `deepmaze-train` Job the Prefect ladder executes.
Defaults to **CPU** (`train_gpu_count=0`) so it runs with no quota; the maze nets
are tiny, so rungs 0–2 converge on CPU. For an **L4**, request the Cloud Run L4
quota (separate from the GCE-T4 quota), then `terraform apply -var train_gpu_count=1`.

```bash
make push-train     # build Dockerfile.train → ghcr.io/juan-garassino/deepmaze-train-job:latest
                    # (flip the GHCR package PUBLIC on first push)
terraform apply -var train_gpu_count=0 \
  -var "wif_pool_id=projects/634336216563/locations/global/workloadIdentityPools/gh-actions"
```

The operator/CI that runs the ladder needs `roles/run.developer` + `actAs` on the
`train_sa_email` output to execute the job. Smoke it on CPU with `make cloudrun-smoke`.

## Show

```bash
cd infra/terraform
terraform init

# One-time: import the shared bucket so TF tracks it without trying to create it.
terraform import google_storage_bucket.artifacts garassino-ml-artifacts

# Push the inference image to GHCR first (from repo root) and flip the
# package to public:
#   make push-backend     # NOT `make push` — that builds the RunPod TRAINING image
# Then apply. Use image_tag=latest for the first apply (deploy.yml takes
# over per-SHA tags on merge; terraform ignores image drift afterwards):
terraform apply \
  -var "image_tag=latest" \
  -var "wif_pool_id=projects/634336216563/locations/global/workloadIdentityPools/gh-actions"
```

> The `wif_pool_id` above is the concrete pool already provisioned in `garassino-op` (project number `634336216563`, pool name `gh-actions`). Discoverable on any machine with gcloud access via:
> `gcloud iam workload-identity-pools list --project=garassino-op --location=global`

Outputs land like:

```
cloud_run_url = "https://deepmaze-backend-<hash>.<region>.run.app"
sa_email      = "deepmaze-backend@garassino-ml.iam.gserviceaccount.com"
bucket_name   = "garassino-ml-artifacts"
assets_prefix = "deepmaze/"
```

Feed those into the GitHub repo secrets/variables per the workspace policy:

| Where | Name | Value |
|---|---|---|
| Secrets | `GCP_PROJECT_ID` | `garassino-ml` |
| Secrets | `CLOUD_RUN_SERVICE` | `deepmaze-backend` |
| Secrets | `CLOUD_RUN_SA_EMAIL` | output `sa_email` |
| Secrets | `WIF_SERVICE_ACCOUNT` | output `sa_email` |
| Secrets | `WIF_PROVIDER` | `<wif_pool_id>/providers/github` (from `garassino-op`) |
| Variables | `GCP_REGION` | `europe-west1` |
| Variables | `ASSETS_BUCKET` | output `bucket_name` |
| Variables | `ASSETS_PREFIX` | output `assets_prefix` |
| Variables | `CORS_ORIGINS` | your frontend origin |

## Destroy

```bash
terraform destroy
```

Tears down the SA, IAM bindings, and Cloud Run service. The **shared bucket survives** (`prevent_destroy=true` on the imported resource) so other deep-* projects' prefixes aren't affected.

## Why `import` and not `create` for the bucket

Workspace convention is "one bucket, per-project prefixes" — created once at the workspace level. Each project's Terraform attaches its own IAM bindings scoped to its prefix but doesn't own the bucket lifecycle. Importing keeps state honest while preserving that contract.

## Cost sanity

After a full apply→destroy cycle, total Cloud Run + IAM cost should land **< €0.50** at €25/mo budget — the only running resource is Cloud Run, which is scale-to-zero. Bucket reads from `sync_assets.py` on cold start are negligible.

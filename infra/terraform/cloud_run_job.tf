// Ephemeral Cloud Run JOB for training (the Prefect ladder executes it once
// per rung; the job trains, syncs artifacts to GCS, and disappears).
//
// GPU: gated on var.train_gpu_count. Default 0 = a CPU job that runs TODAY with
// no quota; set to 1 once the Cloud Run L4 quota lands. torch auto-selects the
// device, so the SAME image serves both. The GPU block shape is verified against
// the google provider docs (template → template → node_selector{accelerator} +
// gpu_zonal_redundancy_disabled + resources.limits["nvidia.com/gpu"]).

locals {
  train_gpu = var.train_gpu_count > 0
  train_limits = local.train_gpu ? {
    cpu              = "4"
    memory           = "16Gi"
    "nvidia.com/gpu" = tostring(var.train_gpu_count)
    } : {
    cpu    = "4"
    memory = "16Gi"
  }
  train_image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.ghcr_remote_repo}/${var.ghcr_owner}/${var.train_image_name}:${var.image_tag}"
}

// ── Runtime SA — writes training artifacts + reads its own warm-start bundle ──
resource "google_service_account" "training" {
  account_id   = "deepmaze-train"
  display_name = "deepMaze training job runtime"
  description  = "Cloud Run Job runtime SA — reads/writes gs://${var.assets_bucket}/deepmaze* (runs workspace + serve prefix)."
}

// objectAdmin (read + write + list) scoped by condition to the `deepmaze` stem,
// which covers both deepmaze-runs/ (workspace) and deepmaze/ (serve). Mirrors
// the two-clause expression backend_reader needs so list_blobs(prefix=...) is
// authorized (object-name clause never matches a list call).
resource "google_storage_bucket_iam_member" "training_rw" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.training.email}"

  condition {
    title       = "deepmaze-stem-only"
    description = "Limit to gs://${var.assets_bucket}/deepmaze* (runs + serve prefixes)."
    expression  = "resource.name.startsWith(\"projects/_/buckets/${var.assets_bucket}/objects/deepmaze\") || api.getAttribute(\"storage.googleapis.com/objectListPrefix\", \"\").startsWith(\"deepmaze\")"
  }
}

// Pull the training image through the ghcr-remote AR repo.
resource "google_project_iam_member" "training_ar_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.training.email}"
}

// Let GitHub Actions / the operator (via garassino-op's WIF pool) impersonate
// the training SA to execute the job.
resource "google_service_account_iam_member" "training_wif_impersonation" {
  service_account_id = google_service_account.training.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${var.wif_pool_id}/attribute.repository/${var.github_repo}"
}

// ── The job ───────────────────────────────────────────────────────────────
resource "google_cloud_run_v2_job" "training" {
  name     = var.train_job_name
  location = var.region

  // Show-and-destroy workspace policy.
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.training.email
      timeout         = var.train_timeout
      max_retries     = 0 // the Prefect ladder owns retries, not Cloud Run

      containers {
        image = local.train_image
        resources {
          limits = local.train_limits
        }

        // Baseline env only; the ladder overrides per rung via
        // `gcloud run jobs update --update-env-vars` (see ignore_changes).
        env {
          name  = "ARTIFACTS_BUCKET"
          value = var.assets_bucket
        }
        env {
          name  = "OUTPUT_BASE"
          value = "/tmp/dm"
        }
      }

      dynamic "node_selector" {
        for_each = local.train_gpu ? [1] : []
        content {
          accelerator = "nvidia-l4"
        }
      }

      gpu_zonal_redundancy_disabled = local.train_gpu ? true : null
    }
  }

  // The ladder mutates per-rung env and `make push-train` rolls the image;
  // Terraform owns the job shell, not its per-run inputs.
  lifecycle {
    ignore_changes = [
      template[0].template[0].containers[0].env,
      template[0].template[0].containers[0].image,
    ]
  }
}

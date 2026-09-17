output "cloud_run_url" {
  description = "Public URL of the deepMaze backend Cloud Run service."
  value       = google_cloud_run_v2_service.backend.uri
}

output "sa_email" {
  description = "Runtime SA email — feed into the WIF_SERVICE_ACCOUNT + CLOUD_RUN_SA_EMAIL GH secrets."
  value       = google_service_account.backend.email
}

output "bucket_name" {
  description = "Shared assets bucket — feed into the ASSETS_BUCKET GH variable."
  value       = google_storage_bucket.artifacts.name
}

output "assets_prefix" {
  description = "Per-project subpath — feed into the ASSETS_PREFIX GH variable."
  value       = var.assets_prefix
}

output "train_job_name" {
  description = "Ephemeral training job — feed into the ladder's TRAIN_JOB env."
  value       = google_cloud_run_v2_job.training.name
}

output "train_sa_email" {
  description = "Training job runtime SA — the operator/CI needs run.developer + actAs on it to execute the job."
  value       = google_service_account.training.email
}

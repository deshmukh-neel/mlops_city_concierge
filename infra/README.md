# Infrastructure (Terraform)

This directory holds Terraform configuration for the City Concierge GCP stack.
The goal of the initial commit series is to bring the **already-running**
infrastructure under Terraform management without recreating any resource —
the exit criterion is `terraform plan` showing zero changes against live state.

## Project

- **GCP Project**: `mlops-491820`
- **Region / Zone**: `us-central1` / `us-central1-a`

## Remote state

State is stored in a GCS bucket with versioning, uniform bucket-level access,
and public-access prevention enforced. Created out-of-band (Terraform can't
bootstrap its own backend) with:

```bash
gcloud storage buckets create gs://mlops-491820-terraform-state \
  --project=mlops-491820 \
  --location=us-central1 \
  --uniform-bucket-level-access \
  --public-access-prevention

gcloud storage buckets update gs://mlops-491820-terraform-state --versioning
```

Verify:

```bash
gcloud storage buckets describe gs://mlops-491820-terraform-state \
  --format="yaml(location,uniform_bucket_level_access,public_access_prevention,versioning_enabled)"
```

Expected: `US-CENTRAL1`, `uniform_bucket_level_access: true`,
`public_access_prevention: enforced`, `versioning_enabled: True`.

## Discovery dumps

Before writing or modifying resource HCL, dump live state to `_discovery/`
(gitignored). The dumps may contain plaintext env vars from Cloud Run — treat
the directory like an SSH key and delete it once a clean plan is achieved.

```bash
mkdir -p _discovery
gcloud sql instances describe mlops--city-concierge \
  --project=mlops-491820 --format=json > _discovery/sql.json
gcloud run services describe city-concierge-api \
  --project=mlops-491820 --region=us-central1 --format=json > _discovery/run.json
gcloud compute instances describe mlflow-server \
  --project=mlops-491820 --zone=us-central1-a --format=json > _discovery/vm.json
gcloud compute firewall-rules describe allow-mlflow \
  --project=mlops-491820 --format=json > _discovery/fw.json
gcloud compute networks describe default \
  --project=mlops-491820 --format=json > _discovery/network.json
```

## Workflow

```bash
cd infra
terraform init                  # backend → gs://mlops-491820-terraform-state
terraform plan -out=tfplan      # NEVER apply during the import exercise
```

Apply is intentionally not run during the import series. The series ends when
`terraform plan` reports `No changes. Your infrastructure matches the
configuration.`

# CI setup for Terraform (Workload Identity Federation)

This runbook configures GitHub Actions to run `terraform plan` on PRs that
touch `infra/**`, authenticating via Workload Identity Federation (no
long-lived service-account keys in GitHub secrets).

**Run these commands once per project.** They are documented here rather
than executed by Terraform because they bootstrap the auth that Terraform
itself depends on — chicken-and-egg.

## Variables used below

| | |
| --- | --- |
| Project ID | `mlops-491820` |
| Project number | `739618408593` |
| Repo | `deshmukh-neel/mlops_city_concierge` |
| WIF pool | `github-actions` |
| WIF provider | `github` |
| Plan-only SA | `terraform-ci@mlops-491820.iam.gserviceaccount.com` |
| State bucket | `mlops-491820-terraform-state` |

## 1. Enable required APIs

```bash
gcloud services enable \
  iamcredentials.googleapis.com \
  iam.googleapis.com \
  sts.googleapis.com \
  --project=mlops-491820
```

## 2. Create the Workload Identity Pool

```bash
gcloud iam workload-identity-pools create github-actions \
  --project=mlops-491820 \
  --location=global \
  --display-name="GitHub Actions"
```

## 3. Create the GitHub OIDC provider in the pool

The `attribute-condition` restricts which GitHub repos can mint tokens
against this provider — without it, *any* GitHub repo could authenticate.

```bash
gcloud iam workload-identity-pools providers create-oidc github \
  --project=mlops-491820 \
  --location=global \
  --workload-identity-pool=github-actions \
  --display-name="GitHub" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository_owner == 'deshmukh-neel'"
```

## 4. Create the plan-only service account

```bash
gcloud iam service-accounts create terraform-ci \
  --project=mlops-491820 \
  --display-name="Terraform CI (plan only)"
```

## 5. Grant the SA the minimum permissions to run `terraform plan`

`plan` only reads — no write permissions on resources. It also needs read
access to the GCS state backend.

```bash
# Read-only on project resources (covers SQL describe, GCE describe, firewall describe)
gcloud projects add-iam-policy-binding mlops-491820 \
  --member="serviceAccount:terraform-ci@mlops-491820.iam.gserviceaccount.com" \
  --role="roles/viewer"

gcloud projects add-iam-policy-binding mlops-491820 \
  --member="serviceAccount:terraform-ci@mlops-491820.iam.gserviceaccount.com" \
  --role="roles/cloudsql.viewer"

# Read+write on the state bucket: terraform plan writes a lockfile in the bucket
# and refreshes state during the plan. objectAdmin is the standard role.
gcloud storage buckets add-iam-policy-binding gs://mlops-491820-terraform-state \
  --member="serviceAccount:terraform-ci@mlops-491820.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

## 6. Bind the GitHub repo to the SA via WIF

This is what lets the GitHub Actions workflow impersonate the SA without
a downloaded key. The `principalSet` matches any token from the
`deshmukh-neel/mlops_city_concierge` repo.

```bash
gcloud iam service-accounts add-iam-policy-binding \
  terraform-ci@mlops-491820.iam.gserviceaccount.com \
  --project=mlops-491820 \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/739618408593/locations/global/workloadIdentityPools/github-actions/attribute.repository/deshmukh-neel/mlops_city_concierge"
```

## 7. Verify

After all of the above, the workflow at `.github/workflows/terraform-plan.yml`
will run on the next PR touching `infra/**` and post a plan comment.

To smoke-test before opening a real change PR: open a no-op PR that adds
a comment to `infra/sql.tf`. The bot should comment with
`No changes. Your infrastructure matches the configuration.`

## Future: apply-on-merge

A separate PR will add a second workflow (`terraform-apply.yml`) that
runs on push to `main` and uses a different SA (`terraform-deploy@`)
with broader permissions. Setup will be similar but with a tighter
`attribute-condition` (e.g. only the `main` branch ref) on either the
provider or the SA binding.

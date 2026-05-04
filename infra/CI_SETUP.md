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

---

# Apply-on-merge setup (terraform-deploy)

The plan workflow above is read-only. To let GitHub Actions actually
mutate GCP on push to `main`, set up a second service account
(`terraform-deploy@`) with `roles/editor`, gated behind:

1. **A manual approval gate** (GitHub Environment `infra-prod`) — the
   workflow pauses before `terraform apply` until a designated reviewer
   approves in the Actions UI.
2. **A `main`-branch-only WIF binding** — the deploy SA can only be
   impersonated from workflow runs triggered by the `main` ref, so a
   malicious PR cannot mint a token that has apply permissions.

## 1. Create the deploy service account

```bash
gcloud iam service-accounts create terraform-deploy \
  --project=mlops-491820 \
  --display-name="Terraform CI (apply)"
```

## 2. Grant project Editor + state bucket admin

```bash
gcloud projects add-iam-policy-binding mlops-491820 \
  --member="serviceAccount:terraform-deploy@mlops-491820.iam.gserviceaccount.com" \
  --role="roles/editor"

gcloud storage buckets add-iam-policy-binding gs://mlops-491820-terraform-state \
  --member="serviceAccount:terraform-deploy@mlops-491820.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

`roles/editor` was chosen for simplicity over scoped IAM
(`cloudsql.admin` + `compute.instanceAdmin.v1` + `compute.securityAdmin`).
The deploy SA is gated behind WIF (only this repo can mint tokens for it)
*and* the GitHub Environment approval gate, so the practical blast
radius is bounded by the human approver, not the IAM scope.

## 3. Create a `main`-only OIDC provider in the same pool

The plan provider (`github`) accepts tokens from any branch — that's
fine for read-only plans. For apply, add a second provider in the same
pool that only accepts tokens whose `ref` claim is `refs/heads/main`.
A malicious PR on a feature branch can't mint a token usable against
the deploy SA, even if it edits the workflow.

```bash
gcloud iam workload-identity-pools providers create-oidc github-main \
  --project=mlops-491820 \
  --location=global \
  --workload-identity-pool=github-actions \
  --display-name="GitHub (main only)" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository_owner == 'deshmukh-neel' && assertion.ref == 'refs/heads/main'"
```

## 4. Bind the deploy SA to `main`-ref tokens only

The principalSet here is keyed on `attribute.ref` (not
`attribute.repository`), so the SA only accepts tokens whose `ref`
claim equals `refs/heads/main`. A token from a feature-branch run
can't satisfy this binding regardless of which provider it came
through.

```bash
gcloud iam service-accounts add-iam-policy-binding \
  terraform-deploy@mlops-491820.iam.gserviceaccount.com \
  --project=mlops-491820 \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/739618408593/locations/global/workloadIdentityPools/github-actions/attribute.ref/refs/heads/main"
```

Defense in depth: the `github-main` OIDC provider's `attribute-condition`
already rejects non-main tokens at provider-level, *and* this SA binding
rejects them at SA-level. Either control on its own is sufficient;
together they fail-safe.

## 5. Configure the GitHub Environment

In the GitHub UI: **Settings → Environments → New environment →
`infra-prod`**.

Configure:
- **Required reviewers**: add the people allowed to approve applies
  (project owners — `pjnhek`, `neel.deshmukh1`, `ankitjai3000`).
- **Deployment branches**: restrict to `main` only.
- (Optional) **Wait timer**: 0 minutes is fine; the human gate is
  the actual control.

The workflow's `environment: infra-prod` line is what triggers the
approval gate.

## 6. Verify

After all of the above:
1. Open a tiny no-op infra PR (e.g. one whitespace change in a `.tf` file).
2. The plan workflow runs and posts a clean plan comment.
3. Merge the PR.
4. The apply workflow starts, reaches the approval step, and pauses.
5. An approver clicks "Review deployments → Approve" in the Actions UI.
6. `terraform apply` runs against `main` and reports the same plan was
   applied.

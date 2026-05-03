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

## Making changes to managed infrastructure

Once the import is in place, **Terraform is the source of truth** for SQL,
the mlflow-server VM, and the allow-mlflow firewall. Do not modify these
in the GCP console or via `gcloud` — your change will be reverted on the
next `terraform apply`, and you'll fight phantom diffs forever.

### Worked example: resize the mlflow-server VM from `e2-small` to `e2-medium`

This is the full loop start-to-finish. Every change to managed infra
follows the same shape — only the file and the field change.

**1. Cut a branch off `main`.** Never edit infra from a branch that also
touches application code.

```bash
git checkout main && git pull
git checkout -b infra/resize-mlflow-vm
```

**2. Edit `infra/compute.tf`.** Find the line you want to change:

```hcl
resource "google_compute_instance" "mlflow_server" {
  name         = "mlflow-server"
  machine_type = "e2-small"   # ← change this to "e2-medium"
  ...
}
```

**3. Format and validate locally.** Catches syntax errors before you
hit the API:

```bash
cd infra
terraform fmt
terraform validate
```

**4. Run `terraform plan` and read it carefully.**

```bash
terraform plan -out=tfplan
```

You should see exactly one diff:

```text
~ resource "google_compute_instance" "mlflow_server" {
    ~ machine_type = "e2-small" -> "e2-medium"
  }

Plan: 0 to add, 1 to change, 0 to destroy.
```

If the plan shows anything else changing — stop. Someone modified the VM
out-of-band via the GCP console or `gcloud`. Investigate before going further.

**5. Commit the `.tf` change** with the plan summary in the message:

```bash
git add compute.tf
git commit -m "infra: resize mlflow-server e2-small -> e2-medium

Plan: 0 to add, 1 to change, 0 to destroy."
git push -u origin infra/resize-mlflow-vm
```

**6. Open a PR.** Paste the full `terraform plan` output into the PR
description so reviewers can see exactly what will hit GCP. Get an approval.

**7. Apply from a clean `main` after merge.** This is when the change
actually goes live:

```bash
git checkout main && git pull
cd infra
terraform plan -out=tfplan      # re-plan to confirm nothing drifted since review
terraform apply tfplan          # this is what mutates GCP
```

`terraform apply tfplan` executes exactly the saved plan — no surprises.
If the re-plan shows a different diff than the PR, stop and investigate.

For VM machine-type changes, GCP requires the instance to be stopped.
Terraform handles this automatically (you'll see `Stopping`, `Modifying`,
`Starting` in the apply output). Expect ~30–60 seconds of MLflow downtime.

**8. Verify the change took effect** in the GCP console (or via `gcloud
compute instances describe mlflow-server --zone=us-central1-a`) and
confirm MLflow is back up. The `.tf` file is already committed — that's
the record of what's running.

### Other common change shapes

- **Resize Cloud SQL disk**: edit `settings.disk_size` in `sql.tf`.
  Disk grows online, no downtime.
- **Bump Cloud SQL tier**: edit `settings.tier` in `sql.tf`. Causes a
  brief restart.
- **Open a new firewall port**: add another `allow { ... }` block to
  the firewall in `compute.tf`, or create a new `google_compute_firewall`
  resource if it's a separate rule.
- **Add a new database flag**: add a `database_flags` block in `sql.tf`.

Same loop in every case: branch → edit → fmt → validate → plan → commit
→ PR → apply from `main`.

### Adding a new resource (not yet under Terraform)

For resources that exist in GCP but aren't yet in Terraform:

1. **Discovery first.** Dump the live resource to `_discovery/` and read
   the JSON before writing any HCL — see the "Discovery dumps" section
   above.
2. **Write the resource block** in the appropriate `.tf` file (or a new
   one — `modules/...` is fine when a logical unit grows past one file).
3. **Add an `import { }` block** in `imports.tf` pointing the resource
   address at the live ID. Use declarative imports, not the `terraform
   import` CLI — they're code-reviewable and survive in git.
4. **Run `terraform plan`.** Iterate on the HCL until the only line in
   the plan summary is `Plan: N to import, 0 to add, 0 to change, 0 to
   destroy.` Resist the urge to "improve" the resource during import —
   parity first, refactors later.
5. **Apply** to commit the import to state, then **delete the
   `import { }` block** in a follow-up commit (it's a one-shot
   instruction, no longer needed once state has the resource).

### What NOT to do

- Edit a managed resource in the GCP console or via `gcloud`. Your change
  will be reverted on the next `terraform apply` and you'll fight phantom
  diffs forever.
- Run `terraform destroy` on the whole config. `deletion_protection` is
  set on Cloud SQL but other resources will go down without warning.
- Commit `_discovery/`, `*.tfstate`, or `tfplan` files. The `.gitignore`
  blocks them; don't `-f` past it.
- Delete or rewrite `.terraform.lock.hcl` casually. It pins provider
  versions across machines; `terraform init -upgrade` is the only
  legitimate way to update it.
- Run `terraform apply` from a feature branch. Always apply from `main`
  after review — that's how the file in git stays the truthful record of
  what's running.

## Resources currently managed

| Resource | Address | Notes |
| --- | --- | --- |
| Cloud SQL instance `mlops--city-concierge` | `google_sql_database_instance.main` | `ignore_changes = [settings[0].maintenance_window]` because live state has `day=0` (any-day) which the provider schema rejects on write |
| GCE VM `mlflow-server` | `google_compute_instance.mlflow_server` | Static internal IP `10.128.0.2`, no public IP |
| Firewall `allow-mlflow` | `google_compute_firewall.allow_mlflow` | TCP/5000 ingress, source `10.128.0.0/9` (overly broad — see cleanup backlog) |

### Final import plan (zero-diff)

```text
Plan: 3 to import, 0 to add, 0 to change, 0 to destroy.
```

## Resources NOT managed (intentional)

- **Cloud Run service `city-concierge-api`** — deployed by GitHub Actions on
  every push. Importing it into Terraform without changing the deploy authority
  would create a permanent two-master problem (TF and CI fighting over every
  revision). Defer until we decide whether TF or CI owns the deploy pipeline.
- **Default VPC and subnets** — auto-created GCP resources, referenced via
  `data` sources rather than imported.

## Follow-up cleanup backlog (separate PRs)

These were observed during discovery but intentionally NOT changed in the
import PR — the import's only job is parity. Each is its own focused change:

1. **Enable Cloud SQL deletion protection** (`settings.deletion_protection_enabled = true`).
   Currently `false`; one bad `gcloud sql instances delete` away from a very bad
   day. Distinct from the resource-level `deletion_protection = true` already
   set in HCL (which only guards `terraform destroy`).
2. **Enable Cloud SQL automated backups** (`settings.backup_configuration.enabled = true`).
   Currently disabled. Combined with #1, this is the most load-bearing fix in
   the backlog.
3. **Remove the `James` (149.36.48.76) authorized network from Cloud SQL.**
   Legacy entry; with `ipv4_enabled = false` it's not reachable anyway, so
   it's dead config — but worth purging.
4. **Tighten `allow-mlflow` firewall source range.** Currently `10.128.0.0/9`
   (basically all RFC1918 10.x); should be the actual subnet that needs to
   reach MLflow (likely `10.128.0.0/20`).
5. **Decide TF-vs-CI ownership for Cloud Run** and either import the service
   with aggressive `ignore_changes` or move deploys behind `terraform apply`.
   Until that decision, Cloud Run stays out of Terraform.
6. **Make `mlflow-server` reproducible.** The VM has no startup script and no
   metadata recorded — MLflow must be running from a manually-installed
   systemd unit. If the VM dies, Terraform recreates an empty Ubuntu box.
   Either capture the install as a startup script or accept that this VM is
   pets, not cattle, and document the bootstrap procedure.

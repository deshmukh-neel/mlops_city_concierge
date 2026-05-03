resource "google_sql_database_instance" "main" {
  # Smoke test for terraform-plan CI workflow (PR #51); revert this comment once verified.
  name             = "mlops--city-concierge"
  database_version = "POSTGRES_18"
  region           = "us-central1"

  # Terraform-side guard against `terraform destroy` (separate from
  # settings.deletion_protection_enabled, which is the GCP-side flag and
  # is currently false in live state — to be enabled in a cleanup PR).
  deletion_protection = true

  settings {
    tier              = "db-perf-optimized-N-8"
    edition           = "ENTERPRISE_PLUS"
    availability_type = "ZONAL"
    activation_policy = "ALWAYS"
    pricing_plan      = "PER_USE"
    disk_type         = "PD_SSD"
    disk_size         = 100
    disk_autoresize   = false

    data_cache_config {
      data_cache_enabled = true
    }

    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }

    backup_configuration {
      enabled                        = false
      point_in_time_recovery_enabled = false
      start_time                     = "17:00"
      transaction_log_retention_days = 14

      backup_retention_settings {
        retained_backups = 15
        retention_unit   = "COUNT"
      }
    }

    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = data.google_compute_network.default.self_link
      ssl_mode                                      = "ALLOW_UNENCRYPTED_AND_ENCRYPTED"
      enable_private_path_for_google_cloud_services = false
      server_ca_mode                                = "GOOGLE_MANAGED_INTERNAL_CA"

      authorized_networks {
        name  = "James"
        value = "149.36.48.76"
      }
    }

    location_preference {
      zone = "us-central1-f"
    }
  }

  # Live state has settings.maintenance_window.day=0 (meaning "any day"),
  # but the Terraform provider's schema rejects day=0 on write. Ignore
  # the field entirely so the imported value sticks without producing a
  # phantom diff every plan.
  lifecycle {
    ignore_changes = [
      settings[0].maintenance_window,
    ]
  }
}

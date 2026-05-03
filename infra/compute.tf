resource "google_compute_instance" "mlflow_server" {
  name         = "mlflow-server"
  machine_type = "e2-small"
  zone         = "us-central1-a"

  tags = ["mlflow-server"]

  boot_disk {
    auto_delete = true
    device_name = "persistent-disk-0"

    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
      size  = 20
      type  = "pd-standard"
    }
  }

  network_interface {
    network    = data.google_compute_network.default.self_link
    subnetwork = data.google_compute_subnetwork.default_us_central1.self_link
    network_ip = "10.128.0.2"
    stack_type = "IPV4_ONLY"
  }

  service_account {
    email = "739618408593-compute@developer.gserviceaccount.com"
    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/pubsub",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append",
    ]
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  can_ip_forward      = false
  deletion_protection = false
}

resource "google_compute_firewall" "allow_mlflow" {
  name        = "allow-mlflow"
  network     = data.google_compute_network.default.self_link
  description = "Allow MLflow server traffic"
  direction   = "INGRESS"
  priority    = 1000
  disabled    = false

  source_ranges = ["10.128.0.0/9"]
  target_tags   = ["mlflow-server"]

  allow {
    protocol = "tcp"
    ports    = ["5000"]
  }
}

data "google_compute_network" "default" {
  name = "default"
}

data "google_compute_subnetwork" "default_us_central1" {
  name   = "default"
  region = "us-central1"
}

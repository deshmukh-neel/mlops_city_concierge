terraform {
  required_version = "~> 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.50"
    }
  }
}

provider "google" {
  project = "mlops-491820"
  region  = "us-central1"
  zone    = "us-central1-a"
}

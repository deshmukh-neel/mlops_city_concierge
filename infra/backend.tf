terraform {
  backend "gcs" {
    bucket = "mlops-491820-terraform-state"
    prefix = "terraform/state"
  }
}

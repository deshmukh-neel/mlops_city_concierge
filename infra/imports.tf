import {
  to = google_sql_database_instance.main
  id = "mlops-491820/mlops--city-concierge"
}

import {
  to = google_compute_instance.mlflow_server
  id = "projects/mlops-491820/zones/us-central1-a/instances/mlflow-server"
}

import {
  to = google_compute_firewall.allow_mlflow
  id = "projects/mlops-491820/global/firewalls/allow-mlflow"
}

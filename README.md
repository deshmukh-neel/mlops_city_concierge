# Team MLflow Tracking Server
Course: USF MSDS 603 - MLOps
Project: City Concierge

## Connection Details
To log experiments to this shared GCP server, use the following tracking URI:
http://35.223.147.177:5000

## How to Log from your Local Environment
Add this block to the top of your Python scripts or Jupyter Notebooks:

```python
import mlflow

# 1. Point to the Team GCP Server
mlflow.set_tracking_uri("http://35.223.147.177:5000")

# 2. Set your experiment name (e.g., 'data-cleaning' or 'model-v1')
mlflow.set_experiment("city-concierge-analysis")

# 3. Start logging!
with mlflow.start_run(run_name="Your_Name_Run"):
    mlflow.log_param("model_type", "regression")
    mlflow.log_metric("rmse", 0.123)
```

## Viewing Results
The live dashboard is accessible at: http://35.223.147.177:5000

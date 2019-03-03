
from __future__ import print_function
import os
import mlflow

def dump_mlflow_info():
  print("MLflow Info:")
  print("  MLflow Version:", mlflow.version.VERSION)
  print("  Tracking URI:", mlflow.tracking.get_tracking_uri())
  print("  MLFLOW_TRACKING_URI:", os.environ.get("MLFLOW_TRACKING_URI",""))
  print("  DATABRICKS_HOST:", os.environ.get("DATABRICKS_HOST",""))
  print("  DATABRICKS_TOKEN:", os.environ.get("DATABRICKS_TOKEN",""))

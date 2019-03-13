
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

from mlflow.utils import databricks_utils

def get_host_(tracking_uri):
    host = os.environ.get('DATABRICKS_HOST',None)
    if host is not None:
        return host
    try:
        db_profile = mlflow.tracking.utils.get_db_profile_from_uri(tracking_uri)
        config = databricks_utils.get_databricks_host_creds(db_profile)
        return config.host
    except Exception as e:
        return None

def get_host(tracking_uri):
    if tracking_uri == "databricks":
        return get_host_(tracking_uri)
    return mlflow.tracking.get_tracking_uri()

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
    mlflow_host = get_mlflow_host_(mlflow.tracking.get_tracking_uri())
    print("  _MLFLOW_HOST:", mlflow_host)

''' Returns the host (tracking URI) and token '''
def get_mlflow_host_token(tracking_uri):
    uri = os.environ.get('MLFLOW_TRACKING_URI',None)
    if uri is not None and uri != "databricks":
        return (uri,None)
    try:
        from mlflow_fun.common import databricks_cli_utils
        profile = os.environ.get('MLFLOW_PROFILE',None)
        host_token = databricks_cli_utils.get_host_token(profile)
        return databricks_cli_utils.get_host_token(profile)
    #except databricks_cli.utils.InvalidConfigurationError as e:
    except Exception as e:
        print("WARNING:",e)
        return (None,None)

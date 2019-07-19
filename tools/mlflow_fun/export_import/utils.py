import mlflow
import os
import time

prefix = "mlflow_tools.export"

def add_log_source_info(client, tags, run):
    uri = mlflow.tracking.get_tracking_uri()
    tags[prefix+".tracking_uri"] = uri
    dbx_host = os.environ.get("DATABRICKS_HOST",None)
    if dbx_host is not None:
        tags[prefix+".DATABRICKS_HOST"] = dbx_host
    now = int(time.time()+.5)
    snow = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))
    tags[prefix+".timestamp"] = now
    tags[prefix+".timestamp_nice"] = snow

    tags[prefix+".run_id"] =  run.info.run_id
    tags[prefix+".experiment_id"] = run.info.experiment_id
    exp = client.get_experiment(run.info.experiment_id)
    tags[prefix+".experiment_name"] = exp.name

def get_now_nice():
    now = int(time.time()+.5)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))

def strip_underscores(obj):
    return { k[1:]:v for (k,v) in obj.__dict__.items() }

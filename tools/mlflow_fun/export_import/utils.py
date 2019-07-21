import os
import json
import time
import mlflow

prefix = "mlflow_tools.export"

# Databricks tags that cannot be set
dbx_skip_tags = set([ "mlflow.user" ])

def create_tags(client, run, log_source_info):
    tags = run.data.tags.copy()
    for tag_key in dbx_skip_tags:
        tags.pop(tag_key, None)

    if not log_source_info:
        return tags

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
    tags[prefix+".user_id"] = run.info.user_id

    return tags

def get_now_nice():
    now = int(time.time()+.5)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))

def strip_underscores(obj):
    return { k[1:]:v for (k,v) in obj.__dict__.items() }

def write_json_file(path, dct):
    with open(path, 'w') as f:
        f.write(json.dumps(dct,indent=2)+"\n")

def read_json_file(path):
    with open(path, "r") as f:
        return json.loads(f.read())

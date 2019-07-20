""" 
Exports a run from one MLflow server and imports it into another server. 
First create an experiment in the destination MLflow server: mlflow experiments create my_experiment
"""

import os
import time
import mlflow
from mlflow_fun.export_import import utils

def get_experiment(client, exp_name):
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = client.create_experiment(exp_name)
        exp = client.get_experiment(exp_id)
    return exp

def run(src_run_id, dst_exp_name, dst_uri, log_source_info):
    print("src_run_id:",src_run_id)
    print("dst_exp_name:",dst_exp_name)
    print("dst_uri:",dst_uri)
    print("log_source_info:",log_source_info)

    src_client = mlflow.tracking.MlflowClient()
    src_uri = mlflow.tracking.get_tracking_uri()
    src_run = src_client.get_run(src_run_id)

    dst_client = mlflow.tracking.MlflowClient(dst_uri)
    dst_exp = get_experiment(dst_client,dst_exp_name)
    print("dst_exp:",dst_exp)
    
    mlflow.tracking.set_tracking_uri(dst_uri)
    with mlflow.start_run(experiment_id=dst_exp.experiment_id) as run:
        dst_run_id = run.info.run_uuid
        print("dst_run_id:",dst_run_id)
        for k,v in src_run.data.params.items():
             mlflow.log_param(k,v)
        for k,v in src_run.data.metrics.items():
             mlflow.log_metric(k,v)
        tags = utils.create_tags(src_client, src_run, log_source_info)
        for k,v in tags.items():
             mlflow.set_tag(k,v)
        # copy artifacts
        local_path = src_client.download_artifacts(src_run_id,"")
        print("local_path:",local_path)
        mlflow.log_artifacts(local_path)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--src_run_id", dest="src_run_id", help="Source run_id", required=True)
    parser.add_argument("--dst_uri", dest="dst_uri", help="Destination MLFLOW API URL", required=True)
    parser.add_argument("--dst_experiment_name", dest="dst_experiment_name", help="Destination experiment_name", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    args = parser.parse_args()
    print("args:",args)
    run(args.src_run_id, args.dst_experiment_name, args.dst_uri, args.log_source_info)

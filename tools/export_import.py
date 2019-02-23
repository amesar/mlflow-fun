
""" 
Exports a run from one MLflow server and imports it into another server. 

First create an experiment in the destination MLflow server: mlflow experiments create my_experiment
"""

from __future__ import print_function

import os
import shutil
from argparse import ArgumentParser
import mlflow
from mlflow import tracking

def run(src_run_id, dst_exp_id, dst_url):
    print("src_run_id:",src_run_id)
    print("dst_exp_id:",dst_exp_id)
    print("dst_url:",dst_url)

    src_client = mlflow.tracking.MlflowClient()
    src_run = src_client.get_run(src_run_id)

    dst_client = mlflow.tracking.MlflowClient(dst_url)
    dst_exp = dst_client.get_experiment(dst_exp_id)
    print("dst_exp:",dst_exp)
    
    mlflow.tracking.set_tracking_uri(dst_url)
    with mlflow.start_run(experiment_id=dst_exp.experiment_id, run_name=src_run.info.name) as run:
        dst_run_id = run.info.run_uuid
        print("dst_run_id:",dst_run_id)
        for e in src_run.data.params:
             mlflow.log_param(e.key, e.value)
        for e in src_run.data.metrics:
             mlflow.log_metric(e.key, e.value)
        for e in src_run.data.tags:
             mlflow.set_tag(e.key, e.value)

    # NOTE: the correct (but slower) way would be to recursively traverse the results of mlflow.get_artifact_uri() 
    local_path = src_client.download_artifacts(src_run_id,"")
    print("local_path:",local_path)
    dst_dir = os.path.join(dst_exp.artifact_location,dst_run_id,"artifacts")
    print("dst_dir:",dst_dir)
    shutil.rmtree(dst_dir)
    shutil.move(local_path, dst_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_run_id", dest="src_run_id", help="Source run_id", required=True)
    parser.add_argument("--dst_url", dest="dst_url", help="Destination API URL", required=True)
    parser.add_argument("--dst_experiment_id", dest="dst_experiment_id", help="Destination experiment_id", required=True, type=int)
    args = parser.parse_args()
    run(args.src_run_id, args.dst_experiment_id, args.dst_url)


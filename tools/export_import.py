
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

def run(src_run_id, dst_exp_id, dst_uri):
    print("src_run_id:",src_run_id)
    print("dst_exp_id:",dst_exp_id)
    print("dst_uri:",dst_uri)

    src_client = mlflow.tracking.MlflowClient()
    src_run = src_client.get_run(src_run_id)

    dst_client = mlflow.tracking.MlflowClient(dst_uri)
    dst_exp = dst_client.get_experiment(dst_exp_id)
    print("dst_exp:",dst_exp)
    
    mlflow.tracking.set_tracking_uri(dst_uri)
    with mlflow.start_run( \
            experiment_id=dst_exp.experiment_id, \
            run_name=src_run.info.name, \
            source_name=src_run.info.source_name, \
            source_version=src_run.info.source_version, \
            source_type=src_run.info.source_type, \
            entry_point_name=src_run.info.entry_point_name \
            # nested=src_run.info.nested # NOTE: where is nested flag?
            ) as run:
        dst_run_id = run.info.run_uuid
        print("dst_run_id:",dst_run_id)
        for e in src_run.data.params:
             mlflow.log_param(e.key, e.value)
        for e in src_run.data.metrics:
             mlflow.log_metric(e.key, e.value)
        for e in src_run.data.tags:
             mlflow.set_tag(e.key, e.value)

        # copy artifacts
        local_path = src_client.download_artifacts(src_run_id,"")
        print("local_path:",local_path)
        mlflow.log_artifacts(local_path)
        shutil.rmtree(local_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_run_id", dest="src_run_id", help="Source run_id", required=True)
    parser.add_argument("--dst_uri", dest="dst_uri", help="Destination API URL", required=True)
    parser.add_argument("--dst_experiment_id", dest="dst_experiment_id", help="Destination experiment_id", required=True, type=int)
    args = parser.parse_args()
    run(args.src_run_id, args.dst_experiment_id, args.dst_uri)


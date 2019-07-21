""" 
Copies a run from one MLflow server to another.
"""

import time
import mlflow
from mlflow_fun.export_import import utils
print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

def setup(dst_exp_name, dst_uri, log_source_info):
    print("Setup:")
    src_client = mlflow.tracking.MlflowClient()
    src_uri = mlflow.tracking.get_tracking_uri()
    print("  src_uri:",src_uri)
    print("  dst_exp_name:",dst_exp_name)
    print("  dst_uri:",dst_uri)
    dst_client = mlflow.tracking.MlflowClient(dst_uri)
    dst_exp = get_experiment(dst_client,dst_exp_name)
    print("  dst_exp.name:",dst_exp.name)
    print("  dst_exp.id:",dst_exp.experiment_id)
    mlflow.tracking.set_tracking_uri(dst_uri)
    print("  log_source_info:",log_source_info)
    return src_client, dst_client, dst_exp

def get_experiment(client, exp_name):
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = client.create_experiment(exp_name)
        exp = client.get_experiment(exp_id)
    return exp

def copy_run(src_run_id, dst_exp_name, dst_uri, log_source_info=False, log_batch=False):
    print("src_run_id:",src_run_id)
    src_client, dst_client, dst_exp = setup(dst_exp_name, dst_uri, log_source_info)
    copy_run2(src_client, src_run_id, dst_client, dst_exp.experiment_id, log_source_info, log_batch)

def copy_run2(src_client, src_run_id, dst_client, dst_experiment_id, log_source_info=False, log_batch=False):
    src_run = src_client.get_run(src_run_id)
    with mlflow.start_run(experiment_id=dst_experiment_id) as dst_run:
        #print("dst_run_id:",dst_run.info.run_uuid)
        if log_batch:
            copy_run_data_batch(src_client, src_run, log_source_info, dst_client, dst_run.info.run_id)
        else:
            copy_run_data(src_client, src_run, log_source_info)
        local_path = src_client.download_artifacts(src_run_id,"")
        #print("local_path:",local_path)
        mlflow.log_artifacts(local_path)

def copy_run_data(src_client, src_run, log_source_info):
    for k,v in src_run.data.params.items():
        mlflow.log_param(k,v)
    for k,v in src_run.data.metrics.items():
        mlflow.log_metric(k,v)
    tags = utils.create_tags(src_client, src_run, log_source_info)
    for k,v in tags.items():
        mlflow.set_tag(k,v)

def copy_run_data_batch(src_client, src_run, log_source_info, dst_client, dst_run_id):
    import time
    from mlflow.entities import Metric, Param, RunTag
    now = int(time.time()+.5)
    params = [ Param(k,v) for k,v in src_run.data.params.items() ]
    metrics = [ Metric(k,v,now,0) for k,v in src_run.data.metrics.items() ] # TODO: check timestamp and step semantics
    tags = utils.create_tags(src_client, src_run, log_source_info)
    tags = [ RunTag(k,v) for k,v in tags.items() ]
    dst_client.log_batch(dst_run_id, metrics, params, tags)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--src_run_id", dest="src_run_id", help="Source run_id", required=True)
    parser.add_argument("--dst_uri", dest="dst_uri", help="Destination MLFLOW API URL", required=True)
    parser.add_argument("--dst_experiment_name", dest="dst_experiment_name", help="Destination experiment_name", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    parser.add_argument("--log_batch", dest="log_batch", help="Use log_batch ", default=False, action='store_true')
    args = parser.parse_args()
    print("args:",args)
    copy_run(args.src_run_id, args.dst_experiment_name, args.dst_uri, args.log_source_info, args.log_batch)

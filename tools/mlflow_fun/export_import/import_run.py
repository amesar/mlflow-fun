"""
Imports a run from a directory of zip file.
"""

import os
import mlflow
from mlflow_fun.export_import import utils

client = mlflow.tracking.MlflowClient()

def import_run(exp_name, input, log_batch=False):
    print("Importing run into experiment '{}' from '{}'".format(exp_name, input),flush=True)
    if input.endswith(".zip"):
        import_run_from_zip(exp_name, input)
    else:
        import_run_from_dir(exp_name, input, log_batch)

def import_run_from_zip(exp_name, zip_file):
    utils.unzip_directory(zip_file, exp_name, import_run_from_dir)

def import_run_from_dir(exp_name, run_dir, log_batch=False):
    mlflow.set_experiment(exp_name)
    exp = client.get_experiment_by_name(exp_name)
    #print("Experiment name:",exp_name)
    #print("Experiment ID:",exp.experiment_id)
    path = os.path.join(run_dir,"run.json")
    run_dct = utils.read_json_file(path)
    with mlflow.start_run() as run:
        if log_batch:
            import_run_data_batch(run_dct,run.info.run_id)
        else:
            import_run_data(run_dct)
        path = os.path.join(run_dir,"artifacts")
        mlflow.log_artifacts(path)

def import_run_data(run_dct):
    for k,v in run_dct['params'].items():
        mlflow.log_param(k,v)
    for k,v in run_dct['metrics'].items():
        mlflow.log_metric(k,v)
    for k,v in run_dct['tags'].items():
        mlflow.set_tag(k,v)

def import_run_data_batch(run_dct, run_id):
    import time
    from mlflow.entities import Metric, Param, RunTag
    now = int(time.time()+.5)
    params = [ Param(k,v) for k,v in run_dct['params'].items() ]
    metrics = [ Metric(k,v,now,0) for k,v in run_dct['metrics'].items() ] # TODO: check timestamp and step semantics
    tags = [ RunTag(k,v) for k,v in run_dct['tags'].items() ]
    client.log_batch(run_id, metrics, params, tags)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input path - directory or zip file", required=True)
    parser.add_argument("--experiment_name", dest="experiment_name", help="Destination experiment_name", required=True)
    parser.add_argument("--log_batch", dest="log_batch", help="Use log_batch ", default=False, action='store_true')
    args = parser.parse_args()
    print("Options:")
    for arg in vars(args):
        print("  {}: {}".format(arg,getattr(args, arg)))
    import_run(args.experiment_name, args.input, args.log_batch)

"""
Imports a run from a directory of zip file.
"""

import os
import json
import tempfile
import shutil
import zipfile
import mlflow

def import_run(exp_name, input):
    print("Importing into {} from {}".format(exp_name, input),flush=True)
    if input.endswith(".zip"):
        import_run_from_zip(exp_name, input)
    else:
        import_run_from_dir(exp_name, input)

def import_run_from_zip(exp_name, zip_file):
    tdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(tdir)
        import_run_from_dir(exp_name, tdir)
    finally:
        shutil.rmtree(tdir)

def import_run_from_dir(exp_name, run_dir):
    mlflow.set_experiment(exp_name)
    path = os.path.join(run_dir,"run.json")
    with open(path, "r") as f:
        run_dct = json.loads(f.read())

    # TODO: use batch API methods
    with mlflow.start_run() as run:
        for k,v in run_dct['params'].items():
            mlflow.log_param(k,v)
        for k,v in run_dct['metrics'].items():
            mlflow.log_metric(k,v)
        for k,v in run_dct['tags'].items():
            mlflow.set_tag(k,v)
        path = os.path.join(run_dir,"artifacts")
        mlflow.log_artifacts(path)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input path - directory or zip file", required=True)
    parser.add_argument("--experiment_name", dest="experiment_name", help="Destination experiment_name", required=True)
    args = parser.parse_args()
    print("args:",args)
    import_run(args.experiment_name, args.input)

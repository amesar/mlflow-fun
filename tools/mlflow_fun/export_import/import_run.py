"""
Imports a run from a directory of zip file.
"""

import os
import json
import mlflow

def import_run(run_dir):
    path = os.path.join(run_dir,"run.json")
    with open(path, "r") as f:
        run_dct = json.loads(f.read())

    # TODO: use batch API methods
    # TODO: import info too
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
    mlflow.set_experiment(args.experiment_name)
    import_run(args.input)

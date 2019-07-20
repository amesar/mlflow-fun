""" 
Exports an experiment to a directory or zip file.
"""

import os
import mlflow
import json
import shutil
import tempfile
from mlflow_fun.export_import import utils, export_run

def export_experiment(exp_id, output, log_source_info=False):
    print("Exporting experiment '{}' to '{}'".format(exp_id,output),flush=True)
    if output.endswith(".zip"):
        export_experiment_from_zip(exp_id, output, log_source_info)
    else:
        os.makedirs(output)
        export_experiment_from_dir(exp_id, output, log_source_info)

def export_experiment_from_dir(exp_id, exp_dir, log_source_info=False):
    client = mlflow.tracking.MlflowClient()
    infos = client.list_run_infos(exp_id)

    exp = client.get_experiment(exp_id)
    path = os.path.join(exp_dir,"manifest.json")
    dct = utils.strip_underscores(exp)
    dct['export_time'] = utils.get_now_nice()
    with open(path, 'w') as f:
          f.write(json.dumps(dct,indent=2)+"\n")

    for j,info in enumerate(infos):
        run_dir = os.path.join(exp_dir, info.run_id)
        print("Exporting run {}/{}: {}".format((j+1),len(infos),info.run_id),flush=True)
        export_run.export_run(info.run_id, run_dir, log_source_info)

def export_experiment_from_zip(exp_id, zip_file, log_source_info=False):
    zip_file,_ = zip_file.split(".")
    tdir = tempfile.mkdtemp()
    try:
        export_experiment_from_dir(exp_id, tdir, log_source_info)
        shutil.make_archive(zip_file, "zip", tdir)
    finally:
        shutil.rmtree(tdir)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_id", dest="experiment_id", help="Source exp_id", required=True)
    parser.add_argument("--output", dest="output", help="Output path", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    args = parser.parse_args()
    print("args:",args)
    export_experiment(args.experiment_id, args.output, args.log_source_info)


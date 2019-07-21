""" 
Exports an experiment to a directory or zip file.
"""

import os
import mlflow
import shutil
import tempfile
from mlflow_fun.export_import import utils, export_run
from mlflow_fun.common import mlflow_utils

client = mlflow.tracking.MlflowClient()

def export_experiment(exp_id_or_name, output, log_source_info=False):
    exp = mlflow_utils.get_experiment(client, exp_id_or_name)
    exp_id = exp.experiment_id


    print("Exporting experiment '{}' (ID {}) to '{}'".format(exp.name,exp.experiment_id,output),flush=True)
    if output.endswith(".zip"):
        export_experiment_to_zip(exp_id, output, log_source_info)
    else:
        os.makedirs(output)
        export_experiment_to_dir(exp_id, output, log_source_info)

def export_experiment_to_dir(exp_id, exp_dir, log_source_info=False):
    exp = client.get_experiment(exp_id)
    dct = {"experiment": utils.strip_underscores(exp)}
    infos = client.list_run_infos(exp_id)
    dct['export_info'] = { 'export_time': utils.get_now_nice(), 'num_runs': len(infos) }
    run_ids = []
    for j,info in enumerate(infos):
        run_dir = os.path.join(exp_dir, info.run_id)
        print("Exporting run {}/{}: {}".format((j+1),len(infos),info.run_id),flush=True)
        export_run.export_run(info.run_id, run_dir, log_source_info)
        run_ids.append(info.run_id)
    dct['run_ids'] = run_ids
    path = os.path.join(exp_dir,"manifest.json")
    utils.write_json_file(path, dct)

def export_experiment_to_zip(exp_id, zip_file, log_source_info=False):
    zip_file,_ = zip_file.split(".")
    tdir = tempfile.mkdtemp()
    try:
        export_experiment_to_dir(exp_id, tdir, log_source_info)
        shutil.make_archive(zip_file, "zip", tdir)
    finally:
        shutil.rmtree(tdir)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment", dest="experiment", help="Source experiment ID or name", required=True)
    parser.add_argument("--output", dest="output", help="Output path", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    args = parser.parse_args()
    print("args:",args)
    export_experiment(args.experiment, args.output, args.log_source_info)

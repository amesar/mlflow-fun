""" 
Exports a run to a directory of zip file.
"""

import os
import shutil
import traceback
import tempfile
import mlflow
from mlflow_fun.export_import import utils
print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

client = mlflow.tracking.MlflowClient()

def export_run(run_id, output, log_source_info=False):
    run = client.get_run(run_id)
    return export_run2(run, output, log_source_info)

def export_run2(run, output, log_source_info=False):
    if output.endswith(".zip"):
        return export_run_to_zip(run, output, log_source_info)
    else:
        os.makedirs(output)
        return export_run_to_dir(run, output, log_source_info)

def export_run_to_zip(run, zip_file, log_source_info=False):
    zip_file,_ = zip_file.split(".")
    tdir = tempfile.mkdtemp()
    try:
        res = export_run_to_dir(run, tdir, log_source_info)
        shutil.make_archive(zip_file, "zip", tdir)
    finally:
        shutil.rmtree(tdir)

def export_run_to_dir(run, run_dir, log_source_info=False):
    dct = { "info": utils.strip_underscores(run.info) , 
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": utils.create_tags(client, run, log_source_info)
          }
    path = os.path.join(run_dir,"run.json")
    utils.write_json_file(path, dct)

    # copy artifacts
    dst_path = os.path.join(run_dir,"artifacts")
    try:
        src_path = client.download_artifacts(run.info.run_id,"")
        shutil.copytree(src_path, dst_path)
        return True
    except Exception as e: # NOTE: Fails for certain runs in Databricks
        print("ERROR: run_id:",run.info.run_id,e)
        traceback.print_exc()
        return False

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="Source run_id", required=True)
    parser.add_argument("--output", dest="output", help="Output directory or zip file", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    args = parser.parse_args()
    print("Options:")
    for arg in vars(args):
        print("  {}: {}".format(arg,getattr(args, arg)))
    export_run(args.run_id, args.output, args.log_source_info)

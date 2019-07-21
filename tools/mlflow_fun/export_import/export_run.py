""" 
Exports a run to a directory of zip file.
"""

import os
import shutil
import tempfile
import mlflow
from mlflow_fun.export_import import utils
print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

client = mlflow.tracking.MlflowClient()

def export_run(run_id, output, log_source_info=False):
    run = client.get_run(run_id)
    export_run2(run, output, log_source_info=False)

def export_run2(run, output, log_source_info=False):
    if output.endswith(".zip"):
        export_run_to_zip(run, output, log_source_info)
    else:
        os.makedirs(output)
        export_run_to_dir(run, output, log_source_info)

def export_run_to_zip(run, zip_file, log_source_info=False):
    zip_file,_ = zip_file.split(".")
    tdir = tempfile.mkdtemp()
    try:
        export_run_to_dir(run, tdir, log_source_info)
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
    #print("dst_path:",dst_path)
    src_path = client.download_artifacts(run.info.run_id,"")
    #print("src_path:",src_path)
    shutil.copytree(src_path, dst_path)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="Source run_id", required=True)
    parser.add_argument("--output", dest="output", help="Output directory or zip file", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    args = parser.parse_args()
    print("args:",args)
    export_run(args.run_id, args.output, args.log_source_info)

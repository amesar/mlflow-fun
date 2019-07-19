""" 
Exports a run to a directory of zip file.
"""

import os
import shutil
import json
import mlflow
from mlflow_fun.export_import import utils

client = mlflow.tracking.MlflowClient()

def export_run(run, run_dir, log_source_info):
    run_id = run.info.run_id
    print("run_id:",run_id)
    print("run_dir:",run_dir)
    print("log_source_info:",log_source_info)

    tags = run.data.tags.copy()
    if log_source_info:
        utils.add_log_source_info(client, tags, run)

    dct = { "info": utils.strip_underscores(run.info) , 
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": tags
          }

    path = os.path.join(run_dir,"run.json")
    with open(path, 'w') as f:
      f.write(json.dumps(dct,indent=2)+'\n')

    # copy artifacts
    src_path = client.download_artifacts(run_id,"")
    print("src_path:",src_path)
    path = os.path.join(run_dir,"artifacts")
    shutil.copytree(src_path,path)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="Source run_id", required=True)
    parser.add_argument("--output", dest="output", help="Output directory or zip file", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    args = parser.parse_args()
    run = client.get_run(args.run_id)
    export_run(run, args.output, args.log_source_info)

""" 
Exports a run to a directory of zip file.
"""

import os
import shutil
import traceback
import tempfile
import mlflow
from mlflow_fun.export_import import utils
from mlflow_fun.common.http_client import DatabricksHttpClient
from mlflow_fun.common import MlflowFunException

print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

dbx_client = DatabricksHttpClient()
print("Databricks REST client:",dbx_client)

client = mlflow.tracking.MlflowClient()

def export_run(run_id, output, log_source_info=False, notebook_formats=["SOURCE"]):
    run = client.get_run(run_id)
    return export_run2(run, output, log_source_info, notebook_formats)

def export_run2(run, output, log_source_info, notebook_formats):
    if output.endswith(".zip"):
        return export_run_to_zip(run, output, log_source_info, notebook_formats)
    else:
        os.makedirs(output)
        return export_run_to_dir(run, output, log_source_info, notebook_formats)

def export_run_to_zip(run, zip_file, log_source_info, notebook_formats):
    temp_dir = tempfile.mkdtemp()
    try:
        res = export_run_to_dir(run, temp_dir, log_source_info, notebook_formats)
        utils.zip_directory(zip_file, temp_dir)
    finally:
        shutil.rmtree(temp_dir)

def export_run_to_dir(run, run_dir, log_source_info, notebook_formats):
    tags =  utils.create_tags(client, run, log_source_info)
    dct = { "info": utils.strip_underscores(run.info) , 
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": tags,
          }
    path = os.path.join(run_dir,"run.json")
    utils.write_json_file(path, dct)

    # copy artifacts
    dst_path = os.path.join(run_dir,"artifacts")
    try:
        src_path = client.download_artifacts(run.info.run_id,"")
        shutil.copytree(src_path, dst_path)
        notebook = tags.get("mlflow.databricks.notebookPath",None)
        if notebook is not None:
            export_notebook(run_dir, notebook, notebook_formats)
        return True
    except Exception as e: # NOTE: Fails for certain runs in Databricks
        print("ERROR: run_id:",run.info.run_id,e)
        traceback.print_exc()
        return False


def export_notebook(run_dir, notebook, notebook_formats):
    for format in notebook_formats:
        _export_notebook(run_dir, notebook, format, format.lower())

def _export_notebook(run_dir, notebook, format, extension):
    resource = f"workspace/export?path={notebook}&direct_download=true&format={format}"
    try:
        rsp = dbx_client._get(resource)
        nb_name = "notebook."+extension
        nb_path = os.path.join(run_dir,nb_name)
        with open(nb_path, 'wb') as f:
            f.write(rsp.content)
    except MlflowFunException as e:
        print(f"WARNING: Cannot save notebook '{notebook}'. {e}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="Source run_id", required=True)
    parser.add_argument("--output", dest="output", help="Output directory or zip file", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    parser.add_argument("--notebook_formats", dest="notebook_formats", default="SOURCE", help="Notebook formats. Values are SOURCE, HTML, JUPYTER, DBC", required=False)
    args = parser.parse_args()
    print("Options:")
    for arg in vars(args):
        print("  {}: {}".format(arg,getattr(args, arg)))
    export_run(args.run_id, args.output, args.log_source_info, utils.string_to_list(args.notebook_formats))

import os
import zipfile
import tempfile
import shutil
import mlflow
from mlflow_fun.export_import import import_run, utils

def import_experiment(exp_name, input):
    if input.endswith(".zip"):
        import_experiment_from_zip(exp_name, input)
    else:
        import_experiment_from_dir(exp_name, input)

def import_experiment_from_dir(exp_name, exp_dir):
    mlflow.set_experiment(exp_name)
    path = os.path.join(exp_dir,"manifest.json")
    dct = utils.read_json_file(path)
    run_dirs = next(os.walk(exp_dir))[1]
    print("Importing {} runs into experiment {} from {}".format(len(run_dirs),exp_name,exp_dir),flush=True)
    for run_dir in run_dirs:
        import_run.import_run(exp_name, os.path.join(exp_dir,run_dir))

def import_experiment_from_zip(exp_name, zip_file):
    tdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(tdir)
        import_experiment_from_dir(exp_name, tdir)
    finally:
        shutil.rmtree(tdir)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="input path", required=True)
    parser.add_argument("--experiment_name", dest="experiment_name", help="Destination experiment_name", required=True)
    args = parser.parse_args()
    print("args:",args)
    import_experiment(args.experiment_name, args.input)

import os
import mlflow
from mlflow_fun.export_import import import_run, utils

def import_experiment(exp_name, input):
    if input.endswith(".zip"):
        import_experiment_from_zip(exp_name, input)
    else:
        import_experiment_from_dir(exp_name, input)

def import_experiment_from_dir(exp_name, exp_dir):
    mlflow.set_experiment(exp_name)
    manifest_path = os.path.join(exp_dir,"manifest.json")
    dct = utils.read_json_file(manifest_path)
    #run_dirs = next(os.walk(exp_dir))[1]
    run_dirs = dct['run_ids']
    failed_run_dirs = dct['failed_run_ids']
    print("Importing {} runs into experiment '{}' from {}".format(len(run_dirs),exp_name,exp_dir),flush=True)
    for run_dir in run_dirs:
        import_run.import_run(exp_name, os.path.join(exp_dir,run_dir))
    print("Imported {} runs into experiment '{}' from {}".format(len(run_dirs),exp_name,exp_dir),flush=True)
    if len(failed_run_dirs) > 0:
        print("Warning: {} failed runs were not imported - see {}".format(len(failed_run_dirs),manifest_path))

def import_experiment_from_zip(exp_name, zip_file):
    utils.unzip_directory(zip_file, exp_name, import_experiment_from_dir)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="input path", required=True)
    parser.add_argument("--experiment_name", dest="experiment_name", help="Destination experiment_name", required=True)
    args = parser.parse_args()
    print("Options:")
    for arg in vars(args):
        print("  {}: {}".format(arg,getattr(args, arg)))
    import_experiment(args.experiment_name, args.input)

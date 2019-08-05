
"""
Recursively dumps all information about an experiment including all details of its runs and their params, metrics and artifacts.
Note that this can be expensive. Adjust your artifact_max_level.
"""

import mlflow
from mlflow_fun.tools import dump_run

client = mlflow.tracking.MlflowClient()
print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

def dump_experiment(exp_id_or_name, artifact_max_level, show_info, show_data):
    print("Options:")
    print("  exp_id_or_name:",exp_id_or_name)
    print("  artifact_max_level:",artifact_max_level)
    print("  show_info:",show_info)
    print("  show_data:",show_data)

    if exp_id_or_name.isdigit():
        exp = client.get_experiment(exp_id_or_name)
        which = "ID"
    else:
        exp = client.get_experiment_by_name(exp_id_or_name)
        which = "name"
    if exp is None:
         raise Exception("Cannot find experiment {} '{}'".format(which,exp_id_or_name))
    exp_id = exp.experiment_id
    print("experiment_id:",exp_id)
    dump_experiment_details(exp)

    if show_info or show_data:
        infos = client.list_run_infos(exp_id)
        print("  #runs:",len(infos))
        #dump_runs(infos, artifact_max_level)
        print("Runs:")
        for j,info in enumerate(infos):
            print("  Run {}/{}:".format(j+1,len(infos)))
            if show_data:
                dump_run.dump_run_id(info.run_uuid, artifact_max_level, indent="    ")
            else:
                dump_run.dump_run_info(info, indent="    ")

def dump_experiment_details(exp):
    print("Experiment Details:")
    for k,v in exp.__dict__.items(): print("  {}: {}".format(k[1:],v))
  
def dump_runs(infos, artifact_max_level):
    print("Runs:")
    for j,info in enumerate(infos):
        print("  Run {}/{}:".format(j+1,len(infos)))
        dump_run.dump_run_id(info.run_uuid, artifact_max_level, indent="    ")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_id_or_name", dest="experiment_id", help="Experiment ID", required=True)
    parser.add_argument("--artifact_max_level", dest="artifact_max_level", help="Number of artifact levels to recurse", required=False, default=1, type=int)
    parser.add_argument("--show_info", dest="show_info", help="Show run info", required=False, default=False, action='store_true')
    parser.add_argument("--show_data", dest="show_data", help="Show run info and data", required=False, default=False, action='store_true')
    args = parser.parse_args()
    print("args:",args)
    dump_experiment(args.experiment_id, args.artifact_max_level,args.show_info, args.show_data)

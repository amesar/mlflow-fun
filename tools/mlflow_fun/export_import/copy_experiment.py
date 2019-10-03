""" 
Copies an experiment from one MLflow server to another.
"""

from mlflow_fun.common import mlflow_utils
from mlflow_fun.export_import import copy_run

def copy_experiment(src_experiment, dst_exp_name, dst_uri, log_source_info=False):
    src_client, dst_client, dst_exp = copy_run.setup(dst_exp_name, dst_uri, log_source_info)
    src_exp = mlflow_utils.get_experiment(src_client, src_experiment)
    print("src_experiment_name:",src_exp.name)
    print("src_experiment_id:",src_exp.experiment_id)
    infos = src_client.list_run_infos(src_exp.experiment_id)
    for j,info in enumerate(infos):
        print("Copying run {}/{}: {}".format((j+1),len(infos),info.run_id),flush=True)
        copy_run.copy_run2(src_client, info.run_id, dst_client, dst_exp.experiment_id, log_source_info)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--src_experiment", dest="src_experiment", help="Source experiment ID or name", required=True)
    parser.add_argument("--dst_uri", dest="dst_uri", help="Destination MLFLOW API URL", required=True)
    parser.add_argument("--dst_experiment_name", dest="dst_experiment_name", help="Destination experiment_name", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    args = parser.parse_args()
    print("args:",args)
    copy_experiment(args.src_experiment, args.dst_experiment_name, args.dst_uri, args.log_source_info)

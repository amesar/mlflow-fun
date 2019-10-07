""" 
Copies an experiment from one MLflow server to another.
"""

import mlflow
from mlflow_fun.common import mlflow_utils
from mlflow_fun.export_import import copy_run
from mlflow_fun.export_import.copy_run import RunCopier
from mlflow_fun.export_import import BaseCopier, create_client

class ExperimentCopier(BaseCopier):

    def __init__(self, src_client, dst_client, log_source_info=False):
        self.log_source_info = log_source_info
        super().__init__(src_client, dst_client, log_source_info)
        self.run_copier = RunCopier(src_client, dst_client, log_source_info)

    def copy_experiment(self, src_exp_id_or_name, dst_exp_name):
        src_exp = mlflow_utils.get_experiment(self.src_client, src_exp_id_or_name)
        dst_exp = self.get_experiment(self.dst_client, dst_exp_name)
        print("src_experiment_name:",src_exp.name)
        print("src_experiment_id:",src_exp.experiment_id)
        infos = self.src_client.list_run_infos(src_exp.experiment_id)
        for j,info in enumerate(infos):
            print("Copying run {}/{}: {}".format((j+1),len(infos),info.run_id),flush=True)
            self.run_copier.copy_run2(info.run_id, dst_exp.experiment_id)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--src_uri", dest="src_uri", help="Source MLFLOW API URL", default=None)
    parser.add_argument("--dst_uri", dest="dst_uri", help="Destination MLFLOW API URL", default=None)
    parser.add_argument("--src_experiment_id_or_name", dest="src_experiment_id_or_name", help="Source experiment ID or name", required=True)
    parser.add_argument("--dst_experiment_name", dest="dst_experiment_name", help="Destination experiment_name", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    args = parser.parse_args()
    print("Options:")
    for arg in vars(args):
        print("  {}: {}".format(arg,getattr(args, arg)))

    src_client = create_client(args.src_uri)
    dst_client = create_client(args.dst_uri)
    print("src_client:",src_client)
    print("dst_client:",dst_client)
    copier = ExperimentCopier(src_client, dst_client, args.log_source_info)
    copier.copy_experiment(args.src_experiment_id_or_name, args.dst_experiment_name)

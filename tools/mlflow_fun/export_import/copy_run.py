""" 
Copies a run from one MLflow server to another.
"""

import time
import mlflow
from mlflow_fun.export_import import utils
from mlflow_fun.export_import import BaseCopier, create_client, add_repr_to_MlflowClient
print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
#add_repr_to_MlflowClient()

class RunCopier(BaseCopier):
    def __init__(self, src_client, dst_client, log_source_info=False):
        super().__init__(src_client, dst_client, log_source_info)

    def copy_run(self, src_run_id, dst_exp_name):
        print("src_run_id:",src_run_id)
        dst_exp = self.get_experiment(self.dst_client,dst_exp_name)
        print("  dst_exp.name:",dst_exp.name)
        print("  dst_exp.id:",dst_exp.experiment_id)
        self.copy_run2(src_run_id, dst_exp.experiment_id)

    def copy_run2(self, src_run_id, dst_experiment_id):
        src_run = self.src_client.get_run(src_run_id)
        dst_run = self.dst_client.create_run(dst_experiment_id) # does not set user_id; is 'unknown'
        dst_run.info.__dict__['_user_id'] = src_run.info.user_id # so we set it to src_run user_id
        self.copy_run_data_batch(src_run, dst_run.info.run_id)
        local_path = self.src_client.download_artifacts(src_run_id,"")
        self.dst_client.log_artifacts(dst_run.info.run_id,local_path)

    def _copy_run2(self, src_run_id, dst_experiment_id):
        src_run = self.src_client.get_run(src_run_id)
        with mlflow.start_run(experiment_id=dst_experiment_id) as dst_run:
            self.copy_run_data_batch(src_run, dst_run.info.run_id)
            local_path = self.src_client.download_artifacts(src_run_id,"")
            #print("local_path:",local_path)
            mlflow.log_artifacts(local_path)

    def copy_run_data_batch(self, src_run, dst_run_id):
        from mlflow.entities import Metric, Param, RunTag
        now = int(time.time()+.5)
        params = [ Param(k,v) for k,v in src_run.data.params.items() ]
        metrics = [ Metric(k,v,now,0) for k,v in src_run.data.metrics.items() ] # TODO: check timestamp and step semantics
        tags = utils.create_tags(self.src_client, src_run, self.log_source_info)
        tags = [ RunTag(k,v) for k,v in tags.items() ]
        self.dst_client.log_batch(dst_run_id, metrics, params, tags)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--src_uri", dest="src_uri", help="Source MLFLOW API URL", default=None)
    parser.add_argument("--dst_uri", dest="dst_uri", help="Destination MLFLOW API URL", default=None)
    parser.add_argument("--src_run_id", dest="src_run_id", help="Source run_id", required=True)
    parser.add_argument("--dst_experiment_name", dest="dst_experiment_name", help="Destination experiment_name", required=True)
    parser.add_argument("--log_source_info", dest="log_source_info", help="Set tags with import information", default=False, action='store_true')
    args = parser.parse_args()
    print("Options:")
    for arg in vars(args):
        print("  {}: {}".format(arg,getattr(args, arg)))
    src_client = create_client(args.src_uri)
    dst_client = create_client(args.dst_uri)
    print("  src_client:",src_client)
    print("  dst_client:",dst_client)
    copier = RunCopier(src_client, dst_client, args.log_source_info)
    copier.copy_run(args.src_run_id, args.dst_experiment_name)

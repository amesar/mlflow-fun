import mlflow

""" 
Client with extra optimized functionality based on search_runs 
"""
class MlflowSmartClient(object):
    def __init__(self, mlflow_client=None, use_fast=True):
        if mlflow_client == None:
            mlflow_client = mlflow.tracking.MlflowClient()
        self.mlflow_client = mlflow_client
        self.use_fast = use_fast

    """ List all run data: info, data.params, data.metrics and data.tags """
    def list_runs(self, experiment_id):
        if self.use_fast:
            return self.list_runs_fast(experiment_id)
        else:
            return self.list_runs_slow(experiment_id)

    def list_runs_fast(self, experiment_id):
        return self.mlflow_client.search_runs([experiment_id], "")

    def list_runs_slow(self, experiment_id):
        infos = self.mlflow_client.list_run_infos(experiment_id)
        return [ self.mlflow_client.get_run(info.run_id) for info in infos ]

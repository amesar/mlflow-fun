from __future__ import print_function
import mlflow
import pandas as pd

''' Client with extra optimized functionality based on search_runs '''
class MlflowSmartClient(object):
    def __init__(self, mlflow_client=None):
        if mlflow_client == None:
            mlflow_client = mlflow.tracking.MlflowClient()
        self.mlflow_client = mlflow_client

    ''' List all run data: info, data.params, data.metrics and data.tags '''
    def list_runs(self, experiment_id):
        return self.mlflow_client.search_runs([experiment_id], "")

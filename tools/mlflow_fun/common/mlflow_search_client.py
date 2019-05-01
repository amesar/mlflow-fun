from __future__ import print_function
import mlflow

''' Client with extra opitmized functionality based on search_runs '''
class MlflowSearchClient(object):
    def __init__(self, mlflow_client=None):
        if mlflow_client == None:
            mlflow_client = mlflow.tracking.MlflowClient()
        self.mlflow_client = mlflow_client

    ''' List all run data: info, data.params, data.metrics and data.tags '''
    def list_runs(self, experiment_id):
        return self.mlflow_client.search_runs([experiment_id], "")

    ''' 
    List all run data as flattened dict. 
    All attributes of info, data.params, data.metrics and data.tags are flattened into one dict.
    Parameters are prefixed with _p_, metrics with _m_ and tags with _t_.
    Example: { "_p_alpha": "0.1", "_m_rmse": 0.82, "_t_data_path": "data.csv", "run_uuid: "123" }
    '''
    def list_runs_flat(self, experiment_id):
        rows = []
        for run in self.list_runs(experiment_id):
            dct = self._strip_underscores(run.info)
            self._merge(dct, run.data.params, '_p_')
            self._merge(dct, run.data.metrics, '_m_')
            self._merge(dct, run.data.tags, '_t_')
            rows.append(dct)
        return rows

    def _merge(self, dct, lst, prefix):
        dct2 = { prefix + x.__dict__['_key'] : x.__dict__['_value'] for x in lst }
        dct.update(dct2)

    def _strip_underscores(self, obj):
        return { k[1:]:v for (k,v) in obj.__dict__.items() }

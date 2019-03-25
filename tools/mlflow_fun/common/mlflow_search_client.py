from __future__ import print_function
from mlflow_fun.common.http_client import HttpClient
from collections import OrderedDict

resource = "runs/search"

''' Client with extra functionality based on REST endpoint runs/search '''
class MlflowSearchClient(object):
    def __init__(self, http_client=None):
        if http_client == None:
            http_client = HttpClient()
        self.http_client = http_client

    ''' Pass JSON search request to REST API '''
    def search(self, request):
        return self.http_client.post(resource,request)

    ''' List all run data: info, data.params, data.metrics and data.tags '''
    def list_runs(self, experiment_id):
        request_template = """ { "experiment_ids": [ {experiment_id} ] } """
        request = request_template.replace("{experiment_id}",str(experiment_id))
        response = self.http_client.post(resource,request)
        return response['runs'] if 'runs' in response else []

    ''' 
    List all run data as flattened dict. Parameters are prefixed with _p_, metrics with _m_ and tags with _t_.
    Example: { "_p_alpha": "0.1", "_m_rmse": 0.82, "_t_data_path": "/dbfs/tmp/data.csv", "run_uuid: "123" }
    '''
    def list_runs_flat(self, experiment_id):
        runs = self.list_runs(experiment_id)
        rows = []
        for run in runs:
            dct = run['info']
            if 'data' in run:
                data = run['data']
                self._merge('params','_p_',dct,data)
                self._merge('metrics','_m_',dct,data)
                self._merge('tags','_t_',dct,data)
            rows.append(dct)
        return rows

    def _merge(self, name, prefix, dct, data):
        if name not in data: return
        lst = data[name]
        dct2 = { prefix+x['key'] : x['value'] for x in lst }
        dct.update(dct2)


import os
import json
import requests
import mlflow

def lt(x,y): return x < y
def gt(x,y): return x > y

def get_best_run_slow(experiment_id, metric, ascending=False):
    funk = lt if ascending else gt
    client = mlflow.tracking.MlflowClient()
    infos = client.list_run_infos(experiment_id)
    best = None
    for info in infos:
        run = client.get_run(info.run_uuid)
        for m in run.data.metrics:
            if m.key == metric:
               if best is None or funk(m.value,best[1]):
                   best = (info.run_uuid,m.value)
    return best

def get_best_run_fast(host, token, experiment_id, metric, ascending=False):
    funk = lt if ascending else gt
    api_path = "api/2.0/preview/mlflow"
    uri = os.path.join(host,api_path,"runs/search")
    req = '{ "experiment_ids": ['+str(experiment_id)+'] }'
    headers = {} if token is None else {'Authorization': 'Bearer '+token}
    rsp = requests.get(uri, data=req, headers=headers)
    runs = json.loads(rsp.text)['runs']
    best = None
    for run in runs:
        if 'data' in run and 'metrics' in run['data']:
            mlist = run['data']['metrics']
            mdct = { x['key'] : x['value'] for x in mlist }
            if metric in mdct:
               mval = mdct[metric]
               if best is None or (isinstance(mval,float) and funk(mval,best[1])):
                  best = (run['info']['run_uuid'],mval)     
    return best

from argparse import ArgumentParser
import mlflow
client = mlflow.tracking.MlflowClient()

def lt(x,y): return x < y
def gt(x,y): return x > y

def calc(metric_name, run, best, funk):
    metric_value = run.data.metrics.get(metric_name,None)
    if metric_value is not None and (best is None or funk(metric_value,best[1])):
         best = (run.info.run_uuid, metric_value)
    return best

"""
Finds the best run by calling get_run for each run. 
"""
def get_best_run_slow(experiment_id, metric, ascending=False):
    funk = lt if ascending else gt
    best = None
    infos = client.list_run_infos(experiment_id)
    for info in infos:
        run = client.get_run(info.run_uuid)
        best = calc(metric, run, best, funk)
    return best

"""
Finds the best run by calling search once to get data for all an experiment's runs
"""
def get_best_run_fast(experiment_id, metric, ascending=False):
    funk = lt if ascending else gt
    best = None
    runs = client.search_runs([experiment_id],"")
    for run in runs:
        best = calc(metric, run, best, funk)
    return best

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_id", dest="experiment_id", help="Experiment ID", type=str, required=True)
    parser.add_argument("--metric", dest="metric", help="Metric", type=str, required=True)
    parser.add_argument("--ascending", dest="ascending", help="ascending", required=False, default=False, action="store_true")
    parser.add_argument("--which", dest="which", help="Which: fast|slow|both", type=str, default="both")
    args = parser.parse_args()
    print("args:",args)
    
    if args.which in ['slow','both']:
        best = get_best_run_slow(args.experiment_id, args.metric, args.ascending)
        print("slow best:",best)
    if args.which in ['fast','both']:
        best = get_best_run_fast(args.experiment_id, args.metric, args.ascending)
        print("fast best:",best)

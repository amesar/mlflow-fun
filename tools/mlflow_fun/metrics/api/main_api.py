from argparse import ArgumentParser
from mlflow_fun.metrics.api.api_best import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", dest="host", help="host", required=True)
    parser.add_argument("--token", dest="token", help="token", required=False, default=None)
    parser.add_argument("--experiment_id", dest="experiment_id", help="Experiment ID", type=int, required=True)
    parser.add_argument("--metric", dest="metric", help="Metric", type=str, required=True)
    parser.add_argument("--ascending", dest="ascending", help="ascending", required=False, default=False, action='store_true')
    args = parser.parse_args()

    best = get_best_run_slow(args.experiment_id, args.metric, args.ascending)
    print("slow best:",best)

    best = get_best_run_fast(args.host, args.token, args.experiment_id, args.metric, args.ascending)
    print("fast best:",best)

from argparse import ArgumentParser
from mlflow_fun.metrics.api.api_best import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_id", dest="experiment_id", help="Experiment ID", type=str, required=True)
    parser.add_argument("--metric", dest="metric", help="Metric", type=str, required=True)
    parser.add_argument("--ascending", dest="ascending", help="ascending", required=False, default=False, action="store_true")
    parser.add_argument("--which", dest="which", help="Which: fast|slow|both", type=str, default="both")
    args = parser.parse_args()

    if args.which in ['slow','both']:
        best = get_best_run_fast(args.experiment_id, args.metric, args.ascending)
        print("fast best:",best)
    if args.which in ['fast','both']:
        best = get_best_run_slow(args.experiment_id, args.metric, args.ascending)
        print("slow best:",best)

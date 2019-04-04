
from argparse import ArgumentParser
from mlflow_fun.metrics.sql.sql_best import get_best_run

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--connection", dest="connection", help="connection", required=True)
    parser.add_argument("--experiment_id", dest="experiment_id", help="experiment_id", type=int, required=True)
    parser.add_argument("--metric", dest="metric", help="metric", required=True)
    parser.add_argument("--ascending", dest="ascending", help="ascending", required=False, default=False, action='store_true')
    args = parser.parse_args()
    best = get_best_run(args.connection, args.experiment_id, args.metric, args.ascending)
    print("best:",best)

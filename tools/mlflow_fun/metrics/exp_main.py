import sys
from mlflow_fun.metrics.table_builder import TableBuilder
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--database", dest="database", help="Database", required=True)
    parser.add_argument("--data_dir", dest="data_dir", help="Data directory", required=True)
    parser.add_argument("--data_frame", dest="data_frame", help="Data_frame: slow or fast", default="slow")
    parser.add_argument("--use_parquet", dest="use_parquet", help="Write as parquet (default CSV)", required=False, default=False, action='store_true')
    parser.add_argument("--experiment_id", dest="experiment_id", help="Experiment ID", type=int, required=True)
    args = parser.parse_args()
    builder = TableBuilder(args.database, args.data_dir, args.data_frame, args.use_parquet)
    builder.build_experiment(args.experiment_id)

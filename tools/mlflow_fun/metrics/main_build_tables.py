import sys
from mlflow_fun.metrics.spark_table_builder import TableBuilder
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--database", dest="database", help="Database", required=True)
    parser.add_argument("--data_dir", dest="data_dir", help="Data directory", required=True)
    parser.add_argument("--use_parquet", dest="use_parquet", help="Write as parquet (default CSV)", required=False, default=False, action='store_true')
    parser.add_argument("--experiment_ids", dest="experiment_ids", help="Experiment IDs", required=False)
    args = parser.parse_args()
    exp_ids = [] if args.experiment_ids is None else [int(id) for id in args.experiment_ids.split(",")]
    builder = TableBuilder(args.database, args.data_dir, args.use_parquet)
    builder.build_experiments(exp_ids)

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("mlflow_metrics").enableHiveSupport().getOrCreate()
    for exp_id in exp_ids:
        table = "{}.exp_{}".format(args.database,exp_id)
        query = "select count(*) from {}".format(table)
        print(query)
        spark.sql(query).show()
        spark.sql("describe table {}".format(table)).show()

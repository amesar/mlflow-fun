import sys
from mlflow_fun.metrics.spark.table_builder import TableBuilder
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--database", dest="database", help="Database", required=True)
    parser.add_argument("--data_dir", dest="data_dir", help="Data directory", required=True)
    parser.add_argument("--use_parquet", dest="use_parquet", help="Write as parquet (default CSV)", required=False, default=False, action='store_true')
    parser.add_argument("--experiment_ids", dest="experiment_ids", help="Experiment IDs", required=False)
    args = parser.parse_args()
    print("Arguments:")
    print("  experiment_ids:",args.experiment_ids)
    print("  database:",args.database)
    print("  data_dir:",args.data_dir)
    print("  use_parquet:",args.use_parquet)

    exp_ids = [] if args.experiment_ids is None else [id for id in args.experiment_ids.split(",")]

    builder = TableBuilder(args.database, args.data_dir, args.use_parquet)
    builder.build_experiments(exp_ids)

#def foo():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("app").enableHiveSupport().getOrCreate()
    print("== SPARK QUERIES:")
    for exp_id in exp_ids:
        table = "{}.exp_{}".format(args.database,exp_id)
        spark.sql("describe table {}".format(table)).show(1000,False)
        query = "select count(*) from {}".format(table)
        print(query)
        spark.sql(query).show(1000,False)

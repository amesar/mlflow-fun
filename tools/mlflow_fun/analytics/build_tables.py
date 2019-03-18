from __future__ import print_function
from pyspark.sql import SparkSession, Row

import sys, os, time
import mlflow
from mlflow_fun.analytics import mlflow_utils
mlflow_utils.dump_mlflow_info()

mlflow_client = mlflow.tracking.MlflowClient()
spark = SparkSession.builder.appName("mlflow_analytics").enableHiveSupport().getOrCreate()

class BuildTables(object):
    def __init__(self, database, data_dir, use_parquet=False, experiment_id=None):
        print("database:",database)
        print("data_dir:",data_dir)
        print("use_parquet:",use_parquet)
        print("experiment_id:",experiment_id)
        self.database = database
        self.data_dir = data_dir
        self.experiment_id = experiment_id
        self.delimiter = "\t"
        self.use_parquet = use_parquet

    # Strip the leading _
    def get_colum_names(self, obj):
        return [x[1:] for x in list(obj.__dict__)]

    def get_values(self, obj):
        values = list(obj.__dict__.values())
        values = [ 0 if x is None else x for x in values ] # TODO: Hack to ensure end_time stays long since a single None makes the column a string ;)
        return values

    def mk_data_path(self,table):
        return self.data_dir + "/" + table

    def mk_fuse_path(self,path):
        return path.replace("dbfs:/","/dbfs/")

    def write_df(self, df, table):
        #df.show()
        #df.printSchema()
        df = df.coalesce(1)
        if self.use_parquet:
            df.write.mode("overwrite").parquet(self.mk_data_path(table))
        else:
            df.write.option("header","true").option("sep",self.delimiter).mode("overwrite").csv(self.mk_data_path(table))

    def build_exp_table_data(self, exps):
        columns = self.get_colum_names(exps[0])
        Experiment = Row(*columns)
        rows = []
        for j,exp in enumerate(exps):
            values = self.get_values(exp)
            rows.append(Experiment(*values))
        df = spark.createDataFrame(rows)
        self.write_df(df,"experiments")

    def build_run_table_data(self, exps):
        Run = None
        rows = []
        for j,exp in enumerate(exps):
            runs = mlflow_client.list_run_infos(exp.experiment_id)
            print("{}/{} Found {} runs for experiment {} - {}".format((1+j),len(exps),len(runs),exp.experiment_id,exp.name))
            for run in runs:
                if Run is None:
                    columns = self.get_colum_names(run)
                    Run = Row(*columns)
                values = self.get_values(run)
                rows.append(Run(*values))
        df = spark.createDataFrame(rows)
        self.write_df(df,"runs")
        print("Total: Found {} experiments and {} runs".format(len(exps),len(rows)))

    def build_table_data(self):
        if self.experiment_id is None:
            exps = mlflow_client.list_experiments() 
        else:
            exps = [ mlflow_client.get_experiment(self.experiment_id) ]
        print("Found {} experiments".format(len(exps)))
        if len(exps) == 0:
            print("WARNING: No experiments found")
            return
        self.build_exp_table_data(exps)
        self.build_run_table_data(exps)

    def build_status_table(self):
        rtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
        print("Refreshed at:",rtime)
        tracking_uri = mlflow.tracking.get_tracking_uri()
        rows = [ Row(refreshed_at=rtime, \
            tracking_uri = tracking_uri,\
            tracking_host = mlflow_utils.get_host(tracking_uri), \
            version = mlflow.version.VERSION) ]
        df = spark.createDataFrame(rows)
        self.write_df(df,"mlflow_status")

    def build_data(self):
        self.mk_dir("experiments")
        self.mk_dir("runs")
        self.mk_dir("mlflow_status")
        self.build_status_table()
        self.build_table_data()

    def build_table_ddl(self, table):
        path = self.mk_data_path(table)
        table = self.database + "." + table
        spark.sql("drop table if exists {}".format(table))
        if self.use_parquet:
            spark.sql('CREATE TABLE {} USING PARQUET\
                OPTIONS (path = "{}") \
                '.format(table,path))
        else:
            spark.sql('CREATE TABLE {} USING CSV\
                OPTIONS (path = "{}", header "true", inferSchema "true", delimiter "{}") \
                '.format(table,path,self.delimiter))

    def build_ddl(self):
        spark.sql("create database if not exists "+self.database)
        self.build_table_ddl("experiments")
        self.build_table_ddl("runs")
        self.build_table_ddl("mlflow_status")

    def mk_dir(self, table):
        path = self.mk_data_path(table)
        path = self.mk_fuse_path(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def build(self):
        self.build_data()
        self.build_ddl()

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--database", dest="database", help="database", required=True)
    parser.add_argument("--data_dir", dest="data_dir", help="data_dir", required=True)
    parser.add_argument("--experiment_id", dest="experiment_id", help="experiment_id", type=int, required=False)
    parser.add_argument("--use_parquet", dest="use_parquet", help="Write as parquet (default CSV)", required=False, default=False, action='store_true')
    args = parser.parse_args()
    builder = BuildTables(args.database, args.data_dir, args.use_parquet, args.experiment_id)
    builder.build()

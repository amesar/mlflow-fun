from pyspark.sql import SparkSession, Row
'''
Builds SQL tables for experiments and runs from the Python API.

Note: In MLflow 1.0.0 user_id exists in RunInfo and also as "mlflow.user" in the tags.
In open source, both are populated. In Databricks managed, only the tag is populated.
Hence the gnarly logic in populating RunInfo.user_id from the tag with idx_user_id etc.
'''

import sys, os, time
from argparse import ArgumentParser
import mlflow
from mlflow_fun.common import mlflow_utils
mlflow_utils.dump_mlflow_info()

mlflow_client = mlflow.tracking.MlflowClient()
spark = SparkSession.builder.appName("mlflow_analytics").enableHiveSupport().getOrCreate()

# RunInfo fields we care about that have been moved over to tags in MLflow 1.0.0
column_tags = { 'source_name': 'mlflow.source.name' }

class BuildTables(object):
    def __init__(self, database, data_dir, use_parquet=False, experiment_ids=None):
        print("database:",database)
        print("data_dir:",data_dir)
        print("use_parquet:",use_parquet)
        print("experiment_ids:",experiment_ids)
        self.database = database
        self.data_dir = data_dir
        self.experiment_ids = experiment_ids
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

    def mk_dir(self, table):
        path = self.mk_data_path(table)
        path = self.mk_fuse_path(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def write_df(self, df, table):
        #df.show()
        #df.printSchema()
        df = df.coalesce(1)
        if self.use_parquet:
            df.write.mode("overwrite").parquet(self.mk_data_path(table))
        else:
            df.write.option("header","true").option("sep",self.delimiter).mode("overwrite").csv(self.mk_data_path(table))

    def build_exp_table(self, exps):
        columns = self.get_colum_names(exps[0])
        Experiment = Row(*columns)
        rows = []
        for j,exp in enumerate(exps):
            values = self.get_values(exp)
            rows.append(Experiment(*values))
        df = spark.createDataFrame(rows)
        self.write_df(df,"experiments")
        self.build_table_ddl("experiments")

    def build_run_table(self, exps):
        run_row = None
        idx_user_id = None
        rows = []
        for j,exp in enumerate(exps):
            try:
                #runs = mlflow_client.search_runs([exp.experiment_id],"") # does not return more than 1000 runs 
                runs = mlflow_client.list_run_infos(exp.experiment_id)
                print("{}/{} Experiment {} has {} runs - {}".format((1+j),len(exps),exp.experiment_id,len(runs),exp.name))
                for info in runs:
                    run = mlflow_client.get_run(info.run_id)
                    if run_row is None:
                        columns = self.get_colum_names(run.info)
                        columns += column_tags.keys()
                        run_row = Row(*columns)
                        idx_user_id = columns.index('user_id')
                    values = self.get_values(run.info)
                    user_id = values[idx_user_id]
                    user_id_exists = user_id is None or user_id == ''
                    if user_id_exists or len(column_tags) > 0:
                        if user_id_exists:
                            mlflow_user = run.data.tags.get('mlflow.user',None)
                            values[idx_user_id] = mlflow_user
                        for col,tag in column_tags.items():
                            v = run.data.tags.get(tag,'')
                            values.append(v)
                    else:
                        user_id = values[idx_user_id] 
                    rows.append(run_row(*values))
            except Exception as e:
                print("WARNING: exp_id:",exp.experiment_id,"ex:",e)
                import traceback
                traceback.print_exc()
        df = spark.createDataFrame(rows)
        self.write_df(df,"runs")
        self.build_table_ddl("runs")
        print("Total: Found {} experiments and {} runs".format(len(exps),len(rows)))


    def build_exp_run_tables(self):
        if self.experiment_ids is None:
            exps = mlflow_client.list_experiments() 
        else:
            exps = [ mlflow_client.get_experiment(exp_id) for exp_id in self.experiment_ids ]
        print("Found {} experiments".format(len(exps)))
        if len(exps) == 0:
            print("WARNING: No experiments found")
            return
        self.build_exp_table(exps)
        self.build_run_table(exps)

    def build_status_table(self):
        rtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
        print("Refreshed at:",rtime)
        tracking_uri = mlflow.tracking.get_tracking_uri()
        rows = [ Row(refreshed_at=rtime, \
            tracking_uri = tracking_uri,\
            tracking_host = mlflow_utils.get_mlflow_host(tracking_uri), \
            version = mlflow.version.VERSION) ]
        df = spark.createDataFrame(rows)
        self.write_df(df,"mlflow_status")
        self.build_table_ddl("mlflow_status")

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
        #print("Table DDL:",table)
        #spark.sql('describe formatted {}'.format(table)).show(100,False)

    def build(self):
        self.mk_dir("experiments")
        self.mk_dir("runs")
        self.mk_dir("mlflow_status")
        spark.sql("create database if not exists "+self.database)
        self.build_status_table()
        self.build_exp_run_tables()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--database", dest="database", help="database", required=True)
    parser.add_argument("--data_dir", dest="data_dir", help="data_dir", required=True)
    parser.add_argument("--experiment_ids", dest="experiment_ids", help="experiment_ids", type=str, required=False)
    parser.add_argument("--use_parquet", dest="use_parquet", help="Write as parquet (default CSV)", required=False, default=False, action='store_true')
    args = parser.parse_args()
    exp_ids = None if args.experiment_ids is None else args.experiment_ids.split(",")
    builder = BuildTables(args.database, args.data_dir, args.use_parquet, exp_ids)
    builder.build()

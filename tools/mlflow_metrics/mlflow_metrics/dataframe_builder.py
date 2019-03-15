from __future__ import print_function
from pyspark.sql import SparkSession, Row

import traceback
import os, time
import mlflow
from mlflow_metrics import mlflow_utils, file_api

class DataframeBuilder(object):
    def __init__(self, spark=None, mlflow_client=None, logmod=20):
        self.logmod = logmod 
        self.mlflow_client = mlflow_client 
        self.spark = spark 
        self.mlflow_client = mlflow_client 
        if mlflow_client is None:
            self.mlflow_client = mlflow.tracking.MlflowClient()
            mlflow_utils.dump_mlflow_info()
        if spark is None:
            self.spark = SparkSession.builder.appName("mlflow_metrics").enableHiveSupport().getOrCreate()
        print("logmod:",logmod)

    def _strip_underscores(self, obj):
        return { k[1:]:v for (k,v) in obj.__dict__.items() }

    def build_dataframe(self, experiment_id, idx=None, num_exps=None):
        (df,n) = self.build_dataframe_(experiment_id, idx, num_exps)
        return df

    def build_dataframe_(self, experiment_id, idx=None, num_exps=None):
        infos = self.mlflow_client.list_run_infos(experiment_id)
        if idx is None:
            print("Experiment {} has {} runs".format(experiment_id,len(infos)))
        else:
            print("{}/{}: Experiment {} has {} runs".format((1+idx),num_exps,experiment_id,len(infos)))
        if len(infos) == 0:
            print("WARNING: No runs for experiment {}".format(exp))
            return (None,0)
        rows = []
        for j,info in enumerate(infos):
            if j%self.logmod==0: print("  run {}/{} of experiment {}".format(j,len(infos),experiment_id))
            run = self.mlflow_client.get_run(info.run_uuid)
            dct = self._strip_underscores(info)
            params = { "_p_"+x.key:x.value for x in run.data.params }
            metrics = { "_m_"+x.key:x.value for x in run.data.metrics }
            dct.update(params)
            dct.update(metrics)
            rows.append(dct)
        df = self.spark.createDataFrame(rows)
        return (df,len(infos))

        #except Exception as e:
            #print("WARNING: Cannot list runs for experiment {} {}".format(experiment_id,e))
            #traceback.print_exc()
            #return 0

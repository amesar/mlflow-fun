# Databricks notebook source
# MAGIC %md dump_experiment.py
# MAGIC * https://github.com/amesar/mlflow-fun/blob/master/tools/mlflow_fun/tools/dump_experiment.py
# MAGIC * synchronized 2019-06-28

# COMMAND ----------

# MAGIC %run ./dump_run.py

# COMMAND ----------

# MAGIC %run ./mlflow_utils

# COMMAND ----------

"""
Recursively dumps all information about an experiment including all details of its runs and their params, metrics and artifacts.
Note that this can be expensive. Adjust your artifact_max_level.
"""

from __future__ import print_function
import mlflow
##from mlflow_fun.tools import dump_run

client = mlflow.tracking.MlflowClient()
print("MLflow Version:", mlflow.version.VERSION)

def dump_experiment(exp_id_or_name, artifact_max_level, show_runs):
    print("Options:")
    print("  exp_id_or_name:",exp_id_or_name)
    print("  artifact_max_level:",artifact_max_level)
    print("  show_runs:",show_runs)
    exp = get_experiment(client, exp_id_or_name)
    exp_id = exp.experiment_id
    print("experiment_id:",exp_id)
    dump_experiment_details(exp)

    if show_runs:
        infos = client.list_run_infos(exp_id)
        print("  #runs:",len(infos))
        dump_runs(infos,artifact_max_level)

def dump_experiment_details(exp):
    print("Experiment Details:")
    for k,v in exp.__dict__.items(): print("  {}: {}".format(k[1:],v))

def dump_runs(infos, artifact_max_level):
    print("Runs:")
    for j,info in enumerate(infos):
        print("  Run {}:".format(j))
        dump_run_id(info.run_uuid, artifact_max_level, indent="    ")
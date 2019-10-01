# Databricks notebook source
# MAGIC %md ### Dump details of all runs of an experiment
# MAGIC * Shows params, metrics and tags
# MAGIC * Recursively shows all artifacts up to the specified level
# MAGIC * Note: Makes lots of calls to API, so beware running with Default experiment 0
# MAGIC 
# MAGIC Links:
# MAGIC * https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.get_run
# MAGIC * https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Run
# MAGIC * https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunData
# MAGIC * https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.list_artifacts

# COMMAND ----------

dbutils.widgets.text(" Experiment ID or name", "1")
dbutils.widgets.dropdown("Show Tags","no",["yes","no"])
dbutils.widgets.dropdown("Show RunInfo Columns","no",["yes","no"])

experiment_id_or_name = dbutils.widgets.get(" Experiment ID or name")
experiment_id = experiment_id_or_name
show_tags = dbutils.widgets.get("Show Tags") == "yes"
show_info_columns = dbutils.widgets.get("Show RunInfo Columns") == "yes"

# COMMAND ----------

from mlflow_fun.tools.pandas_utils import *
df = create_pandas_dataframe(experiment_id, True, True, show_tags, show_info_columns, False)
display(df)

# COMMAND ----------

type(df)
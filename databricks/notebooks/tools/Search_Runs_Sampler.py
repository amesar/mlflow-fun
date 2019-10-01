# Databricks notebook source
# MAGIC %md ## Explore new search runs features
# MAGIC 
# MAGIC There are two search runs methods in the Python API:
# MAGIC   * mlflow.search_runs - returns PandasDataFrame
# MAGIC   * mlflow.tracking.MlflowClient.search_runs - returns MLflow [Run](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Run) entity
# MAGIC 
# MAGIC Documentation:
# MAGIC   * https://mlflow.org/docs/latest/search-syntax.html
# MAGIC   * https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs
# MAGIC   * https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_runs

# COMMAND ----------

# MAGIC %md #### Setup

# COMMAND ----------

import mlflow
print("MLflow Version:", mlflow.version.VERSION)
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# If experiment doesn't yet exist

with mlflow.start_run() as run:
  mlflow.log_param("dummy","dummy")

# COMMAND ----------

experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
experiment_id, experiment_name

# COMMAND ----------

# MAGIC %md #### Create Runs

# COMMAND ----------

for run in client.search_runs(experiment_id):
    client.delete_run(run.info.run_id)

# COMMAND ----------

num_runs = 10
for j in range(0,num_runs):
    with mlflow.start_run() as run:
        mlflow.log_param("p",f"p{j}")
        mlflow.log_metric("rmse", round(.7+.01*j,2))
        mlflow.set_tag("algo", ("dt" if j % 2 == 0 else "lr"))

# COMMAND ----------

# Create last run

import time
time.sleep(1)
with mlflow.start_run() as run:
  mlflow.log_param("p","last_run")
  mlflow.log_metric("rmse",.9)

# COMMAND ----------

# MAGIC %md #### Search Runs - All

# COMMAND ----------

pdf = mlflow.search_runs(experiment_id)
display(pdf)

# COMMAND ----------

runs = client.search_runs(experiment_id)
display(runs)

# COMMAND ----------

# MAGIC %md #### Search Runs - Latest run

# COMMAND ----------

order_by = ["attributes.start_time desc"]

# COMMAND ----------

pdf = mlflow.search_runs(experiment_id, order_by=order_by, max_results=1)
display(pdf)

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=order_by, max_results=1)
runs[0]

# COMMAND ----------

# MAGIC %md #### Search Runs - Best run

# COMMAND ----------

order_by = ["metrics.rmse asc"]

# COMMAND ----------

pdf = mlflow.search_runs(experiment_id,"", order_by=order_by, max_results=1)
display(pdf)

# COMMAND ----------

runs = client.search_runs(experiment_id,"", order_by=order_by, max_results=1)
runs[0]

# COMMAND ----------

# MAGIC %md #### Search Runs - query

# COMMAND ----------

order_by = ["metrics.rmse"]
query = "metrics.rmse > .75"

# COMMAND ----------

pdf = mlflow.search_runs(experiment_id, query, order_by=order_by)
display(pdf)

# COMMAND ----------

runs = client.search_runs(experiment_id, query, order_by=order_by)
for run in runs:
    print(run.info.run_id,run.data.params['p'],run.data.metrics['rmse'])

# COMMAND ----------

# MAGIC %md #### Search Runs - max_results

# COMMAND ----------

# MAGIC %md ##### Max Results - mlflow.search_runs
# MAGIC 
# MAGIC https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs
# MAGIC * max_results – The maximum number of runs to put in the dataframe. Default is 100,000 to avoid causing out-of-memory issues on the user’s machine.
# MAGIC 
# MAGIC Notes: 
# MAGIC * No paging as in mlflow.tracking.MlflowClient.search_runs
# MAGIC * Doesn't seem to honor 100,000 as I can pass it 1,000,000 - see below.

# COMMAND ----------

runs = mlflow.search_runs(experiment_id, max_results=1000000)
display(runs)

# COMMAND ----------

# MAGIC %md ##### Max Results - mlflow.tracking.MlflowClient.search_runs
# MAGIC 
# MAGIC https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_runs
# MAGIC * max_results – Maximum number of runs desired. `NOTE: no mention of maximum value`
# MAGIC 
# MAGIC Exception: 
# MAGIC * Databricks: INVALID_PARAMETER_VALUE: Argument max_results should be <= 50000
# MAGIC * Open source: Invalid value for request parameter max_results. It must be at most 50000, but got value 1000000

# COMMAND ----------

try:
  client.search_runs(experiment_id, max_results=1000000)
except Exception as e:
  print(e)

# COMMAND ----------

# MAGIC %md #### Search Runs - Iterator - advanced
# MAGIC 
# MAGIC The number of results of the search_runs method is controlled by the max_results argument.

# COMMAND ----------

# MAGIC %run ../common/search_runs_iterator.py

# COMMAND ----------

it = SearchRunsIterator(client, experiment_id)
for run in it:
    print(run.info.run_id,run.data.params['p'],run.data.metrics['rmse'])

# COMMAND ----------

max_results = 50000
it = SearchRunsIterator(client, experiment_id, max_results)
for run in it:
    print(run.info.run_id,run.data.params['p'],run.data.metrics['rmse'])

# COMMAND ----------

it = SearchRunsIterator(client, experiment_id, query="metrics.rmse > .75")
for run in it:
    print(run.info.run_id,run.data.params['p'],run.data.metrics['rmse'])
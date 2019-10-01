# Databricks notebook source
# MAGIC %md ## Search Runs

# COMMAND ----------

dbutils.widgets.text("Experiment ID", "") 
dbutils.widgets.text("Query", "") 

exp_id = dbutils.widgets.get("Experiment ID")
query = dbutils.widgets.get("Query")
exp_id, query

# COMMAND ----------

import mlflow
print("MLflow Version:", mlflow.version.VERSION)
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

runs = client.search_runs([exp_id], query)

# COMMAND ----------

type(runs)

# COMMAND ----------

for run in runs:
  #print("type:",type(run))
  print("====",run.info.run_id+"\n")
  print(run)
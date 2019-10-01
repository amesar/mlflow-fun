# Databricks notebook source
# MAGIC %md ### List nested run IDs of a parent run

# COMMAND ----------

dbutils.widgets.text("Run ID", "")
run_id = dbutils.widgets.get("Run ID")
run_id

# COMMAND ----------

import mlflow
print("MLflow Version:", mlflow.version.VERSION)
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

def list_child_run_ids(parent_run_id):
    print("parent_run_id:",parent_run_id)
    run = client.get_run(parent_run_id)
    print("experiment_id:",run.info.experiment_id)
    runs = client.search_runs([run.info.experiment_id], "")
    child_run_ids = []
    for run in runs:
        tags = { tag.key:tag.value for tag in run.data.tags }
        child_run_id = tags.get("mlflow.parentRunId",None)
        if child_run_id is not None and child_run_id == parent_run_id:
            child_run_ids.append(child_run_id)
    return child_run_ids

# COMMAND ----------

list_child_run_ids(run_id)
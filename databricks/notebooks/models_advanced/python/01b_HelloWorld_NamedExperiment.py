# Databricks notebook source
# MAGIC %md ## Hello World with Experiment Name
# MAGIC * Libraries:
# MAGIC   * Attach PyPi mlflow library to your cluster
# MAGIC * Experiment: [/Shared/experiments/Default Experiment](https://demo.cloud.databricks.com/#mlflow/experiments/2100777)

# COMMAND ----------

import mlflow

experiment_name = "/Shared/experiments/Default Experiment"
print("experiment_name:", experiment_name)
mlflow.set_experiment(experiment_name)
print("MLflow Version:", mlflow.version.VERSION)

# COMMAND ----------

def make_run_uri(run):
  host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName")
  return "https://{}/#mlflow/experiments/{}/runs/{}".format(host_name.get(),run.info.experiment_id,run.info.run_uuid)

with mlflow.start_run(run_name='HelloWorld') as run:
    print("run_id:",run.info.run_uuid)
    print("experiment_id:",run.info.experiment_id)
    print("artifact_uri:",mlflow.get_artifact_uri())
    print("run uri:",make_run_uri(run))
    mlflow.log_param("alpha", "0.1")
    mlflow.log_metric("rmse", 0.876)
    mlflow.set_tag("algorithm", "hello_world")
    with open("info.txt", "w") as f:
        f.write("My artifact")
    mlflow.log_artifact("info.txt")
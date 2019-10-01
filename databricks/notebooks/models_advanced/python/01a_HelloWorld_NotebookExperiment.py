# Databricks notebook source
# MAGIC %md ## Hello World with Notebook Experiment
# MAGIC * Libraries:
# MAGIC   * Attach PyPi mlflow library to your cluster
# MAGIC * Notes:
# MAGIC   * After running, check the "Runs" button in the upper left menu bar for your results

# COMMAND ----------

import mlflow
print("MLflow :", mlflow.version.VERSION)
client = mlflow.tracking.MlflowClient()
#notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
#print("notebook_path:  ",notebook_path)

def make_run_uri(run):
  host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName")
  return "https://{}/#mlflow/experiments/{}/runs/{}".format(host_name.get(),run.info.experiment_id,run.info.run_uuid)

with mlflow.start_run() as run:
    print("run_id:",run.info.run_uuid)
    print("experiment_id:",run.info.experiment_id)
    experiment = client.get_experiment(run.info.experiment_id)
    print("experiment_name:",experiment.name)
    print("artifact_uri:",mlflow.get_artifact_uri())
    print("run uri:",make_run_uri(run))
    mlflow.log_param("alpha", "0.1")
    mlflow.log_metric("rmse", 0.876)
    mlflow.set_tag("algorithm", "hello_world")
    with open("info.txt", "w") as f:
        f.write("My artifact")
    mlflow.log_artifact("info.txt")
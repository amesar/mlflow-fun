# Databricks notebook source
# MAGIC %md ## Hello World with Notebook Experiment
# MAGIC * Demonstrates basic MLflow API methods - no model training.
# MAGIC * Libraries:
# MAGIC   * Attach PyPi mlflow library to your cluster.
# MAGIC * Notes:
# MAGIC   * After running, check the "Runs" button in the upper right menu bar for your results.

# COMMAND ----------

import mlflow
print("MLflow :", mlflow.version.VERSION)

with mlflow.start_run() as run:
    print("run_id:",run.info.run_uuid)
    print("experiment_id:",run.info.experiment_id)
    print("artifact_uri:",mlflow.get_artifact_uri())
    mlflow.log_param("alpha", "0.5")
    # Train model
    mlflow.log_metric("rmse", 0.876)
    mlflow.set_tag("algorithm", "hello_world")
    with open("info.txt", "w") as f:
        f.write("My artifact")
    mlflow.log_artifact("info.txt")

# COMMAND ----------

def display_run_uri(experiment_id, run_id):
    host_name = dbutils.notebook.entry_point.getDbutils() \
        .notebook().getContext().tags() \
        .get("browserHostName").get()
    uri = "https://{}/#mlflow/experiments/{}/runs/{}".format(host_name,experiment_id,run_id)
    displayHTML("""<b>Run URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

display_run_uri(run.info.experiment_id, run.info.run_id)

# COMMAND ----------


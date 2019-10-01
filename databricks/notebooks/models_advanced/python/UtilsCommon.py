# Databricks notebook source
# MAGIC %run ../../tools/dump_run.py

# COMMAND ----------

import mlflow
print("MLflow Version:", mlflow.version.VERSION)
mlflow_client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# If running from within notebook.run(), host_name will be unknown
def create_run_uri(experiment_id, run_id):
    host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName")
    host_name = "_?_" if host_name.isEmpty() else host_name.get()
    return "https://{}/#mlflow/experiments/{}/runs/{}".format(host_name,experiment_id,run_id)

# COMMAND ----------

# Utility function to display run URI
# Example: https://demo.cloud.databricks.com/#mlflow/experiments/2580010/runs/5cacf3cbad89413d8167cfe54eaec8dd

def display_run_uri(experiment_id, run_id):
    uri = create_run_uri(experiment_id, run_id)
    displayHTML("""Run URI: <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def mk_dbfs_path(fuse_path):
    return fuse_path.replace("/dbfs","dbfs:")

# COMMAND ----------

def get_username():
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
  
def get_home_dir():
  return "/Users/{}".format(get_username())

# COMMAND ----------

def remove_widget(name):
    try:
        dbutils.widgets.remove(name)
    except Exception as e: # com.databricks.dbutils_v1.InputWidgetNotDefined
        pass

# COMMAND ----------

EXPERIMENT_NAME_PREFIX = "glr"
def create_experiment_name(name, prefix=EXPERIMENT_NAME_PREFIX):
  return "{}/{}_{}".format(get_home_dir(),prefix,name)

# COMMAND ----------

import os
import requests

# Example:
#  data_path = "/dbfs/tmp/mlflow_wine-quality.csv"
#  data_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"

def download_file(data_uri, data_path):
    if os.path.exists(data_path):
        print("File {} already exists".format(data_path))
    else:
        print("Downloading {} to {}".format(data_uri,data_path))
        rsp = requests.get(data_uri)
        with open(data_path, 'w') as f:
            f.write(requests.get(data_uri).text)

# COMMAND ----------

def download_wine_file():
    data_path = "/dbfs/tmp/mlflow_wine-quality.csv" # TODO: per username
    data_uri = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
    download_file(data_uri, data_path)
    return data_path

# COMMAND ----------

def read_wine_data():
    data_path = download_wine_file()
    return spark.read.format("csv") \
      .option("header", "true") \
      .option("inferSchema", "true") \
      .load(data_path.replace("/dbfs","dbfs:")) 

# COMMAND ----------

colLabel = "quality"
colPrediction = "prediction"
colFeatures = "features"
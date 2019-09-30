# Databricks notebook source
import os
import requests

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
    data_path = "/dbfs/tmp/mlflow_wine-quality.csv"
    data_uri = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
    download_file(data_uri, data_path)
    return data_path

# COMMAND ----------

def display_run_uri(experiment_id, run_id):
    host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
    uri = "https://{}/#mlflow/experiments/{}/runs/{}".format(host_name,experiment_id,run_id)
    displayHTML("""<b>Run URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def to_int(x):
  return None if x is None or x=="" else int(x)
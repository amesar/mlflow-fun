# Databricks notebook source
# MAGIC %md ## Save features as tag and search
# MAGIC * Save a list of features as a JSON string in a tag
# MAGIC * Search an experiment for a specific feature list

# COMMAND ----------

import json
import mlflow
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# MAGIC %md #### Create runs with features

# COMMAND ----------

def train(features):
    with mlflow.start_run() as run:
        mlflow.log_param("max_depth","10")
        mlflow.set_tag("features",json.dumps(features))

# COMMAND ----------

features_ab = ["a","b"]
features_xy = ["x","y"]

# COMMAND ----------

train(features_ab)
train(features_ab)
train(features_xy)

# COMMAND ----------

# MAGIC %md #### Print all runs details

# COMMAND ----------

experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

# COMMAND ----------

runs = client.search_runs(experiment_id)
for run in runs: print("{} {}".format(run.info.run_id,run.data.tags['features']))

# COMMAND ----------

# MAGIC %md #### Search for features

# COMMAND ----------

def search_for_features(features):
    query = "tags.features = '{}'".format(json.dumps(features))
    print("Query:",query)
    return [ (run.info.run_id,run.data.tags['features']) \
       for run in client.search_runs(experiment_id,query) ]   

# COMMAND ----------

search_for_features(features_ab)

# COMMAND ----------

search_for_features(features_xy)
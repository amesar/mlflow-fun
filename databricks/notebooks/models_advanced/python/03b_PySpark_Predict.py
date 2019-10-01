# Databricks notebook source
# MAGIC %md ## Predict PySpark Model 
# MAGIC * Runs predictions for a run from [03a_PySpark_Train](https://demo.cloud.databricks.com/#notebook/3538051).
# MAGIC * Reads flavors: Spark ML and MLeap.

# COMMAND ----------

# MAGIC %run ./predict_widgets_description.txt

# COMMAND ----------

# MAGIC %md ### Includes

# COMMAND ----------

# MAGIC %run ./UtilsPredict

# COMMAND ----------

# MAGIC %md ### Setup widgets

# COMMAND ----------

#remove_predict_widgets()

# COMMAND ----------

exp_name = create_experiment_name("pyspark")
run_id,experiment_id,experiment_name,run_id,metric,run_mode = create_predict_widgets(exp_name,["mae","r2","rmse"])
run_id,experiment_id,experiment_name,metric,run_mode

# COMMAND ----------

dump_run_id(run_id)

# COMMAND ----------

# MAGIC %md ### Read Data

# COMMAND ----------

data = read_wine_data()

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

# MAGIC %md #### Predict using Spark ML format

# COMMAND ----------

import mlflow.spark

model_uri = "runs:/{}/spark-model".format(run_id)
model = mlflow.spark.load_model(model_uri)
predictions = model.transform(data)
display(predictions.select(colPrediction, colLabel, colFeatures))

# COMMAND ----------

# MAGIC %md #### Predict using MLeap format

# COMMAND ----------

import mlflow

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
run.info.artifact_uri

# COMMAND ----------

mleap_path = "{}/mleap-model/mleap/model".format(run.info.artifact_uri)
mleap_path = mleap_path.replace("dbfs:","/dbfs")
bundle_path = "file:" + mleap_path
bundle_path

# COMMAND ----------

from pyspark.ml import PipelineModel
from mleap.pyspark.spark_support import SimpleSparkSerializer

model = PipelineModel.deserializeFromBundle(bundle_path)
predictions = model.transform(data)
display(predictions.select(colPrediction, colLabel, colFeatures))

# COMMAND ----------

# MAGIC %md #### Return

# COMMAND ----------

dbutils.notebook.exit(run_id)
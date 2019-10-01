# Databricks notebook source
# MAGIC %md ## Predict Keras/Tensorflow Model with UDF
# MAGIC * Runs predictions for a run from [Train_Keras_TF_Housing](https://demo.cloud.databricks.com/#notebook/2966731).
# MAGIC * Reads following flavors:
# MAGIC   * UDF - [mlflow.pyfunc.spark_udf()](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf)
# MAGIC * Libraries:
# MAGIC   * PyPI package: mlflow 
# MAGIC * Includes:
# MAGIC   * UtilsPredict

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

import keras
print("Keras Version:", keras.__version__)
import tensorflow
print("TensorFlow Version:", tensorflow.__version__)

# COMMAND ----------

# MAGIC %run ./UtilsPredict

# COMMAND ----------

# MAGIC %md ### Setup widgets

# COMMAND ----------

metrics = ['val_loss',
 'val_mean_squared_error',
 'val_mean_absolute_error',
 'loss',
 'mean_squared_error',
 'mean_absolute_error']

exp_name = create_experiment_name("keras")
run_id,experiment_id,experiment_name,run_id,metric,run_mode = create_predict_widgets(exp_name,metrics)

run_id,experiment_id,experiment_name,run_id,metric,run_mode,exp_name

# COMMAND ----------

dump_run_id(run_id)

# COMMAND ----------

# MAGIC %md ### Read data

# COMMAND ----------

from sklearn.datasets.california_housing import fetch_california_housing
data = fetch_california_housing().data

# COMMAND ----------

# MAGIC %md ### Review the MLflow UI

# COMMAND ----------

display_run_uri(experiment_id, run_id)

# COMMAND ----------

# MAGIC %md ### Predict
# MAGIC 
# MAGIC Let's now register our Keras model as a Spark UDF to apply to rows in parallel.

# COMMAND ----------

# MAGIC %md #### DataFrame UDF

# COMMAND ----------

import pandas as pd
import mlflow.pyfunc
model_uri = "runs:/"+run_id+"/keras-model"
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
df = spark.createDataFrame(pd.DataFrame(data))
display(df.withColumn("prediction", udf("0", "1", "2", "3", "4", "5", "6", "7")))

# COMMAND ----------

# MAGIC %md #### SQL UDF

# COMMAND ----------

spark.udf.register("predictUDF", udf)
df.createOrReplaceGlobalTempView("data")

# COMMAND ----------

# MAGIC %sql
# MAGIC select *, predictUDF(*) as prediction from global_temp.data

# COMMAND ----------

dbutils.notebook.exit(run_id)
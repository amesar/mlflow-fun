# Databricks notebook source
# MAGIC %md ## MLflow Advanced Sklearn Predict
# MAGIC * Runs predictions for runs from [02a_Sklearn_Train](https://demo.cloud.databricks.com/#notebook/3528347).
# MAGIC * Predicts in several different ways:
# MAGIC   * sklearn - [mlflow.sklearn.load_model()](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.load_model)
# MAGIC   * UDF (pyfunc) - [mlflow.pyfunc.spark_udf()](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf)
# MAGIC   * pyfunc - [mlflow.pyfunc.load_pyfunc()](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_pyfunc)

# COMMAND ----------

# MAGIC %run ./predict_widgets_description.txt

# COMMAND ----------

# MAGIC %md ### Includes

# COMMAND ----------

# MAGIC %run ./UtilsPredict

# COMMAND ----------

# MAGIC %md ### Setup widgets

# COMMAND ----------

exp_name = create_experiment_name("sklearn")
run_id,experiment_id,experiment_name,run_id,metric,run_mode = create_predict_widgets(exp_name, ["mae","r2","rmse"])
run_id,experiment_id,experiment_name,metric,run_mode

# COMMAND ----------

# MAGIC %md ### Dump Run

# COMMAND ----------

dump_run_id(run_id)

# COMMAND ----------

# MAGIC %md ### Read data

# COMMAND ----------

data_path = "dbfs:/tmp/mlflow_wine-quality.csv"
df = spark.read.option("inferSchema",True).option("header", True).csv(data_path)
df = df.drop("quality")

# COMMAND ----------

# MAGIC %md ### Utility function
# MAGIC Concatenate panda predictions and data as Spark Dataframe

# COMMAND ----------

import pandas as pd

def concat_predictions(predictions, pandas_df):
    df_pred = pd.DataFrame(predictions, columns=["Prediction"])
    df_full = pd.concat([df_pred,pandas_df], axis=1)
    return spark.createDataFrame(df_full)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = "runs:/{}/sklearn-model".format(run_id)
model_uri

# COMMAND ----------

# MAGIC %md #### Predict with mlflow.pyfunc.spark_udf

# COMMAND ----------

# MAGIC %md ##### DataFrame UDF

# COMMAND ----------

import mlflow.pyfunc

udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns))
display(predictions)

# COMMAND ----------

# MAGIC %md ##### SQL UDF

# COMMAND ----------

spark.udf.register("predictUDF", udf)
df.createOrReplaceGlobalTempView("data")

# COMMAND ----------

# MAGIC %sql
# MAGIC select *, predictUDF(*) as prediction from global_temp.data

# COMMAND ----------

# MAGIC %md #### Predict with mlflow.sklearn.load_model

# COMMAND ----------

import mlflow.sklearn

model = mlflow.sklearn.load_model(model_uri)
pandas_df = df.toPandas()
predictions = model.predict(pandas_df)
display(concat_predictions(predictions, pandas_df))

# COMMAND ----------

# MAGIC %md #### Predict with mlflow.pyfunc.load_pyfunc

# COMMAND ----------

mlflow_client = mlflow.tracking.MlflowClient()

model_uri = mlflow_client.get_run(run_id).info.artifact_uri + "/sklearn-model"
model_uri = model_uri.replace("dbfs:","/dbfs")
model = mlflow.pyfunc.load_pyfunc(model_uri)

predictions = model.predict(pandas_df)
display(concat_predictions(predictions, pandas_df))

# COMMAND ----------

dbutils.notebook.exit(run_id)
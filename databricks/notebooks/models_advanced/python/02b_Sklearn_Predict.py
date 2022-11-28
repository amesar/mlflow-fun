# Databricks notebook source
# MAGIC %md ## MLflow Advanced Sklearn Predict
# MAGIC * Runs predictions for runs from [02a_Sklearn_Train](https://demo.cloud.databricks.com/#notebook/3528347).
# MAGIC * Predicts with following flavors:
# MAGIC   * sklearn - [mlflow.sklearn.load_model()](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.load_model)
# MAGIC   * UDF - [mlflow.pyfunc.spark_udf()](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf) - as DataFrame and SQL
# MAGIC   * pyfunc - both klearn and ONX - [mlflow.pyfunc.load_pyfunc()](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_pyfunc)
# MAGIC   * ONNX - [mlflow.onnx.load_model](https://www.mlflow.org/docs/latest/python_api/mlflow.onnx.html#mlflow.onnx.load_model)

# COMMAND ----------

# MAGIC %run ./predict_widgets_description.txt

# COMMAND ----------

# MAGIC %md ### Pip install libraries

# COMMAND ----------

# MAGIC %pip install onnx==1.9.0
# MAGIC %pip install onnxmltools==1.7.0
# MAGIC %pip install onnxruntime==1.8.0
# MAGIC %pip install mlflow==1.19.0

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

data_path = mk_dbfs_path(download_wine_file())
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

model_uri = f"runs:/{run_id}/sklearn-model"
model_uri

# COMMAND ----------

# MAGIC %md #### Predict as UDF

# COMMAND ----------

# MAGIC %md ##### DataFrame UDF

# COMMAND ----------

import mlflow.pyfunc

udf = mlflow.pyfunc.spark_udf(spark, model_uri)

# COMMAND ----------

predictions = df.withColumn("prediction", udf(*df.columns))
type(predictions), predictions.count(), len(predictions.columns)

# COMMAND ----------

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

# MAGIC %md #### Predict with Sklearn flavor

# COMMAND ----------

import mlflow.sklearn
model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

pandas_df = df.toPandas()
predictions = model.predict(pandas_df)
type(predictions), predictions.shape

# COMMAND ----------

display(concat_predictions(predictions, pandas_df))

# COMMAND ----------

# MAGIC %md #### Predict as Pyfunc/Sklearn flavor

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(pandas_df)
type(predictions), predictions.shape

# COMMAND ----------

#display(concat_predictions(predictions, pandas_df))
display(pd.DataFrame(predictions, columns=["prediction"]))

# COMMAND ----------

# MAGIC %md ### Predict as ONNX

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
artifacts = client.list_artifacts(run_id, "onnx-model")
if len(artifacts) == 0:
    print("Exiting since there is no ONNX flavor")
    dbutils.notebook.exit(run_id)

# COMMAND ----------

# MAGIC %md #### Predict as Pyfunc/ONNX flavor

# COMMAND ----------

model_uri = f"runs:/{run_id}/onnx-model"
model_uri

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(pandas_df)
type(predictions), predictions.shape

# COMMAND ----------

display(predictions)

# COMMAND ----------

# MAGIC %md #### Predict as direct ONNX flavor

# COMMAND ----------

import mlflow.onnx
import onnxruntime
import numpy as np

model = mlflow.onnx.load_model(model_uri)
sess = onnxruntime.InferenceSession(model.SerializeToString())
input_name = sess.get_inputs()[0].name
predictions = sess.run(None, {input_name: pandas_df.to_numpy().astype(np.float32)})[0]
type(predictions), predictions.shape

# COMMAND ----------

display(pd.DataFrame(predictions, columns=["prediction"]))

# COMMAND ----------

# MAGIC %md ## Return

# COMMAND ----------

dbutils.notebook.exit(run_id)
# Databricks notebook source
# MAGIC %md # Basic Sklearn MLflow train and predict
# MAGIC * Trains and saves model as sklearn
# MAGIC * Predicts using Sklearn and PyFunc UDF flavors
# MAGIC * Option to save model signature

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

# Default values per: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

dbutils.widgets.text("Experiment Name","")
dbutils.widgets.text("Registered Model","")
dbutils.widgets.dropdown("Save Signature","no",["yes","no"])
dbutils.widgets.text("Max Depth", "1") 
dbutils.widgets.text("Max Leaf Nodes", "")
max_depth = to_int(dbutils.widgets.get("Max Depth"))

max_leaf_nodes = to_int(dbutils.widgets.get("Max Leaf Nodes"))
save_signature = dbutils.widgets.get("Save Signature") == "yes"
experiment_name = dbutils.widgets.get("Experiment Name")
registered_model = dbutils.widgets.get("Registered Model")
if registered_model=="": registered_model = None

print("experiment_name:",experiment_name)
print("registered_model:",registered_model)
print("max_depth:",max_depth)
print("max_leaf_nodes:",max_leaf_nodes)
print("save_signature:",save_signature)

# COMMAND ----------

import sklearn
import mlflow
import mlflow.sklearn

# COMMAND ----------

if not experiment_name == "":
    mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data_path = download_wine_file()

# COMMAND ----------

import pandas as pd
data = pd.read_csv(data_path)
display(data)

# COMMAND ----------

data.describe()

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.30, random_state=42)

# The predicted column is colLabel which is a scalar from [3, 9]
train_x = train.drop([colLabel], axis=1)
test_x = test.drop([colLabel], axis=1)
train_y = train[colLabel]
test_y = test[colLabel]

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

# COMMAND ----------

model_name = "sklearn-model"

with mlflow.start_run(run_name="sklearn") as run:
    run_id = run.info.run_uuid
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",run.info.experiment_id)
    print("Parameters:")
    print("  max_depth:",max_depth)
    print("  max_leaf_nodes:",max_leaf_nodes)
    
    mlflow.set_tag("version.mlflow", mlflow.__version__)
    mlflow.set_tag("save_signature", save_signature)

    model = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x, train_y)
      
    predictions = model.predict(test_x)
    signature = infer_signature(train_x, predictions) if save_signature else None
    print("signature:",signature)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_leaf_nodes", max_leaf_nodes)
        
    mlflow.sklearn.log_model(model, "sklearn-model", signature=signature, registered_model_name=registered_model)
        
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    r2 = r2_score(test_y, predictions)
    print("Metrics:")
    print("  rmse:",rmse)
    print("  r2:",r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2) 

# COMMAND ----------

display_run_uri(run.info.experiment_id, run_id)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = f"runs:/{run_id}/{model_name}"
model_uri

# COMMAND ----------

# MAGIC %md #### Predict as sklearn

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
data_to_predict = data.drop(colLabel, axis=1)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions,columns=[colPrediction]))

# COMMAND ----------

type(predictions)

# COMMAND ----------

# MAGIC %md #### Predict as PyFunc

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions,columns=[colPrediction]))

# COMMAND ----------

type(predictions)

# COMMAND ----------

# MAGIC %md #### Predict as UDF

# COMMAND ----------

df = spark.createDataFrame(data_to_predict)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns))
display(predictions)

# COMMAND ----------

type(predictions)
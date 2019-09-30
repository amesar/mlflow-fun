# Databricks notebook source
# MAGIC %md # Basic Sklearn MLflow train and predict notebook
# MAGIC * Trains and saves model as sklearn
# MAGIC * Predicts using sklearn and pyfunc UDF flavors

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

metrics = ["rmse","r2"]
colLabel = "quality"
colPrediction = "prediction"
colFeatures = "features"

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

# Default values per: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

dbutils.widgets.text("maxDepth", "1") 
dbutils.widgets.text("maxLeafNodes", "")
max_depth = to_int(dbutils.widgets.get("maxDepth"))
max_leaf_nodes = to_int(dbutils.widgets.get("maxLeafNodes"))
max_depth, max_leaf_nodes

# COMMAND ----------

import mlflow
import mlflow.sklearn
print("MLflow Version:", mlflow.version.VERSION)

# COMMAND ----------

metrics = ["rmse","r2"]
colLabel = "quality"
colPrediction = "prediction"
colFeatures = "features"

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

train, test = train_test_split(data, test_size=0.30, random_state=2019)

# The predicted column is colLabel which is a scalar from [3, 9]
train_x = train.drop([colLabel], axis=1)
test_x = test.drop([colLabel], axis=1)
train_y = train[[colLabel]]
test_y = test[[colLabel]]

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn

# COMMAND ----------

with mlflow.start_run() as run:
    run_id = run.info.run_uuid
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",run.info.experiment_id)

    dt = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    dt.fit(train_x, train_y)
    y_predict = dt.predict(test_x)
    predictions = y_predict

    print("Parameters:")
    print("  max_depth:",max_depth)
    print("  max_leaf_nodes:",max_leaf_nodes)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_leaf_nodes", max_leaf_nodes)

    mlflow.sklearn.log_model(dt, "sklearn-model")

    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
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

model_uri = "runs:/{}/sklearn-model".format(run_id)

# COMMAND ----------

# MAGIC %md #### Predict as sklearn

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
data_to_predict = data.drop(colLabel, axis=1)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions,columns=[colPrediction]))

# COMMAND ----------

# MAGIC %md #### Predict as UDF

# COMMAND ----------

df = spark.createDataFrame(data_to_predict)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns))
display(predictions)
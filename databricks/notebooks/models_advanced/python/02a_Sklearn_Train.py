# Databricks notebook source
# MAGIC %md ## MLflow Advanced Sklearn Train Model
# MAGIC 
# MAGIC * Libraries:
# MAGIC   * PyPI package: mlflow 
# MAGIC * Synopsis:
# MAGIC   * Model: DecisionTreeRegressor
# MAGIC   * Data: Wine quality
# MAGIC   * Logs  model as sklearn (pickle) format
# MAGIC   * Logs a PNG plot artifact

# COMMAND ----------

# MAGIC %md ### Pip install libraries

# COMMAND ----------

# MAGIC %pip install onnx==1.9.0
# MAGIC %pip install onnxmltools==1.7.0
# MAGIC %pip install onnxruntime==1.8.0

# COMMAND ----------

# MAGIC %run ../../Versions

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./UtilsTrain

# COMMAND ----------

# Default values per: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

dbutils.widgets.text(WIDGET_TRAIN_EXPERIMENT, create_experiment_name("sklearn"))
dbutils.widgets.text("Max Depth", "") 
dbutils.widgets.text("Max Leaf Nodes", "")
dbutils.widgets.dropdown("Use ONNX","no",["yes","no"])

experiment_name = dbutils.widgets.get(WIDGET_TRAIN_EXPERIMENT)   
max_depth = to_int(dbutils.widgets.get("Max Depth"))
max_leaf_nodes = to_int(dbutils.widgets.get("Max Leaf Nodes"))
log_as_onnx = dbutils.widgets.get("Use ONNX") == "yes"

max_depth, max_leaf_nodes, experiment_name, log_as_onnx

# COMMAND ----------

set_experiment(experiment_name)

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

train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# COMMAND ----------

# MAGIC %md ### Plot

# COMMAND ----------

import matplotlib.pyplot as plt

def create_plot_file(y_test_set, y_predicted, plot_file):
    global image
    fig, ax = plt.subplots()
    ax.scatter(y_test_set, y_predicted, edgecolors=(0, 0, 0))
    ax.plot([y_test_set.min(), y_test_set.max()], [y_test_set.min(), y_test_set.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Ground Truth vs Predicted")
    image = fig
    fig.savefig(plot_file)
    plt.close(fig)

# COMMAND ----------

# MAGIC %md ### Train
# MAGIC 
# MAGIC * ONNX note: 1.8.0 - ConnectException error: This is often caused by an OOM error that causes the connection to the Python REPL to be closed. Check your query's memory usage.

# COMMAND ----------

import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from mlflow.entities import Metric, Param, RunTag

with mlflow.start_run() as run:
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    print("MLflow:")
    print("  run_id:", run_id)
    print("  experiment_id:", experiment_id)  

    #  Params
    print("Parameters:")
    print("  max_depth:", max_depth)
    print("  max_leaf_nodes:", max_leaf_nodes)
    
    # Create and fit model
    dt = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    dt.fit(train_x, train_y)
        
    # Metrics
    predictions = dt.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)  
    print("Metrics:")
    print("  rmse:", rmse)
    print("  mae:", mae)
    print("  r2:", r2)
    
    # Log params, metrics and tags
    import time
    now = round(time.time())
    metrics = [Metric("rmse",rmse, now, 0), Metric("r2", r2, now, 0)]
    params = [Param("max_depth", str(max_depth)), 
              Param("max_leaf_nodes", str(max_leaf_nodes))]
    client.log_batch(run_id, metrics, params, create_version_tags())

    # Log Sklearn model
    mlflow.sklearn.log_model(dt, "sklearn-model")
    
    # Log ONNX model
    if log_as_onnx:
        import mlflow.onnx
        import onnx
        import skl2onnx
        initial_type = [('float_input', skl2onnx.common.data_types.FloatTensorType([None, test_x.shape[1]]))]
        onnx_model = skl2onnx.convert_sklearn(dt, initial_types=initial_type)
        print("onnx_model.type:", type(onnx_model))
        mlflow.onnx.log_model(onnx_model, "onnx-model")
        mlflow.set_tag("onnx_version", onnx.__version__)
    
    # Create and log plot
    plot_file = "ground_truth_vs_predicted.png"
    create_plot_file(test_y, predictions, plot_file)
    mlflow.log_artifact(plot_file)

# COMMAND ----------

# MAGIC %md ### Display run

# COMMAND ----------

display_run_uri(experiment_id, run_id)

# COMMAND ----------

dump_run_id(run_id, 5)

# COMMAND ----------

# MAGIC %md ### Return result

# COMMAND ----------

dbutils.notebook.exit(f"{run_id} {experiment_id}")
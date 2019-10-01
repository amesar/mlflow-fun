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

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./UtilsTrain

# COMMAND ----------

# Default values per: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

dbutils.widgets.text(WIDGET_TRAIN_EXPERIMENT, create_experiment_name("sklearn"))
dbutils.widgets.text("maxDepth", "") 
dbutils.widgets.text("maxLeafNodes", "")

experiment_name = dbutils.widgets.get(WIDGET_TRAIN_EXPERIMENT)   
max_depth = to_int(dbutils.widgets.get("maxDepth"))
max_leaf_nodes = to_int(dbutils.widgets.get("maxLeafNodes"))

max_depth, max_leaf_nodes, experiment_name

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

# COMMAND ----------

import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

with mlflow.start_run() as run:
    mlflow.log_param("mlflow.version",mlflow.version.VERSION)
    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",experiment_id)  

    # Log params
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_leaf_nodes", max_leaf_nodes)
    print("Parameters:")
    print("  max_depth:",max_depth)
    print("  max_leaf_nodes:",max_leaf_nodes)
    
    # Create and fit model
    dt = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    dt.fit(train_x, train_y)
        
    # Log metrics
    predictions = dt.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)  
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    print("Metrics:")
    print("  rmse:",rmse)
    print("  mae:",mae)
    print("  r2:",r2)

    # Log model
    mlflow.sklearn.log_model(dt, "sklearn-model")
    
    # Create and log plot
    plot_file = "ground_truth_vs_predicted.png"
    create_plot_file(test_y, predictions, plot_file)
    mlflow.log_artifact(plot_file)

# COMMAND ----------

# MAGIC %md ### Display run

# COMMAND ----------

display_run_uri(experiment_id, run_id)

# COMMAND ----------

dump_run_id(run_id)

# COMMAND ----------

# MAGIC %md ### Return result

# COMMAND ----------

dbutils.notebook.exit("{} {}".format(run_id,experiment_id))
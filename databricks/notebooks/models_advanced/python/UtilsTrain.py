# Databricks notebook source
# MAGIC %run ./UtilsCommon

# COMMAND ----------

WIDGET_TRAIN_EXPERIMENT = " Experiment Name"

# COMMAND ----------

def set_experiment(experiment_name):
    print("experiment_name:",experiment_name)
    if experiment_name == "":
        experiment_id = None
    else:
        mlflow.set_experiment(experiment_name)
        experiment_id = mlflow_client.get_experiment_by_name(experiment_name).experiment_id
    print("experiment_id:",experiment_id)
    return experiment_id

# COMMAND ----------

remove_widget(WIDGET_TRAIN_EXPERIMENT)

# COMMAND ----------

def remove_train_widgets():
    remove_widget(WIDGET_TRAIN_EXPERIMENT)
    #remove_widget("Experiment")
    #remove_widget(" Experiment")
    remove_widget("maxDepth")
    remove_widget("maxBins")
    remove_widget("maxLeafNodes")
remove_train_widgets()

# COMMAND ----------

def to_int(x):
  return None if x is None or x=="" else int(x)
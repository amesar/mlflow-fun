# Databricks notebook source
# MAGIC %run ./UtilsCommon

# COMMAND ----------

# MAGIC %run ../../common/Best_Run_Utils

# COMMAND ----------

# %run ./PySpark_UDF_Workaround

# COMMAND ----------

def create_train_notebook_path():
  npath = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
  return npath.replace("Predict_","Train_")

# COMMAND ----------

#WIDGET_PREDICT_EXPERIMENT = "Experiment ID or Name"
WIDGET_PREDICT_EXPERIMENT = "Experiment"
default_run_id = ""
RUN_MODE, BEST_RUN, LAST_RUN, RUN_ID = " Run Mode", "Best Run", "Last Run", "Run ID"

def create_predict_widgets(exp_name, metrics):
    print("exp_name:",exp_name)
    dbutils.widgets.text(WIDGET_PREDICT_EXPERIMENT,exp_name)
    dbutils.widgets.dropdown("Metric Name",metrics[0], metrics)
    dbutils.widgets.dropdown("Metric Sort","min",["min","max"])
    dbutils.widgets.text("Run ID",default_run_id)

    dbutils.widgets.dropdown(RUN_MODE, LAST_RUN,[RUN_ID,BEST_RUN,LAST_RUN])
    exp_id_or_name = dbutils.widgets.get(WIDGET_PREDICT_EXPERIMENT)
    print("exp_id_or_name:",exp_id_or_name)
    exp_id,exp_name = get_experiment_info(exp_id_or_name)
    print("exp_id:",exp_id)
    print("exp_name:",exp_name)
    
    run_mode = dbutils.widgets.get(RUN_MODE)
    metric = ""
    if run_mode == RUN_ID:
        run_id = dbutils.widgets.get("Run ID")
    elif run_mode == LAST_RUN:
        run = get_last_parent_run(exp_id)
        run_id = run.info.run_uuid
    elif run_mode == BEST_RUN:
        metric = dbutils.widgets.get("Metric Name")
        ascending = dbutils.widgets.get("Metric Sort")
        best = get_best_run_fast(exp_id,metric,ascending=="min")
        print("best:",best,"metric:",metric)
        run_id = best[0]
    else:
        run_id = dbutils.widgets.get("Run ID") # TODO: Error

    return run_id,exp_id,exp_name,run_id,metric,run_mode

# COMMAND ----------

def remove_predict_widgets():
    remove_widget(" Run Mode")
    remove_widget("Experiment ID or Name")
    remove_widget(" Experiment")
    remove_widget("Metric Name")
    remove_widget("Metric Sort")
    remove_widget("Run ID")

# COMMAND ----------

def get_experiment_id(exp_id_or_name):
  if exp_id_or_name.isdigit(): 
      return exp_id_or_name
  exp = mlflow_client.get_experiment_by_name(exp_id_or_name)
  return exp.experiment_id

# COMMAND ----------

def get_experiment_info(exp_id_or_name):
  if exp_id_or_name.isdigit(): 
      exp = mlflow_client.get_experiment(exp_id_or_name)
      exp_name = None if exp is None else exp.name
      return (exp_id_or_name,exp_name)
  else:
      exp = mlflow_client.get_experiment_by_name(exp_id_or_name)
      exp_id = None if exp is None else exp.experiment_id
      return (exp_id,exp_id_or_name)
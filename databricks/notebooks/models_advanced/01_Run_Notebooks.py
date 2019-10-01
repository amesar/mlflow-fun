# Databricks notebook source
# MAGIC %md ## Run All MLflow Model Advanced Gallery`Notebooks
# MAGIC 
# MAGIC Overview
# MAGIC * Automated testing of Advanced Gallery notebooks.
# MAGIC * Calls train and predict notebooks for each model.
# MAGIC * For prediction, runs in RUN_ID mode and passes the run_id result of the training notebook.
# MAGIC 
# MAGIC Widgets
# MAGIC * Experiment prefix - Prefix for experiments which will be found in the user's home directory default is `glr_run_`.
# MAGIC * Home dir - If blank will use user's home directory 
# MAGIC * Delete runs - delete all runs from experiments before running.
# MAGIC * Train many - run several training runs with different parameters for selected experiments.
# MAGIC 
# MAGIC After running these xperiments will be in your home directory
# MAGIC * glr_run_sklearn
# MAGIC * glr_run_pyspark
# MAGIC * glr_run_pyspark_trackMLlib
# MAGIC * glr_run_keras
# MAGIC * glr_run_scala

# COMMAND ----------

# MAGIC %md ## Initialization

# COMMAND ----------

dbutils.widgets.text(" Experiment prefix","glr_run")
dbutils.widgets.text(" Home dir","")
dbutils.widgets.dropdown("Train many","no",["yes","no"])
dbutils.widgets.dropdown("Delete runs","no",["yes","no"])
dbutils.widgets.dropdown("Skip failed tests","no",["yes","no"])

train_many = dbutils.widgets.get("Train many") == "yes"
delete_runs = dbutils.widgets.get("Delete runs") == "yes"
skipped_failed_tests = dbutils.widgets.get("Skip failed tests") == "yes"
experiment_prefix = dbutils.widgets.get(" Experiment prefix")
home_dir = dbutils.widgets.get(" Home dir")

experiment_prefix,home_dir,train_many,delete_runs,skipped_failed_tests

# COMMAND ----------

# MAGIC %md ### RunManager Setup

# COMMAND ----------

# MAGIC %run ./Run_Manager

# COMMAND ----------

exp_names = [ "sklearn", "pyspark", "pyspark_trackMLlib", "keras", "scala" ]
if home_dir == "":
    home_dir = "/Users/"+dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
exp_name_base = "{}/{}_".format(home_dir,experiment_prefix)
home_dir, exp_name_base

# COMMAND ----------

run_manager = RunManager(exp_names, exp_name_base)
run_manager.list_runs()

# COMMAND ----------

if delete_runs:
    run_manager.delete_runs()
    run_manager.list_runs()

# COMMAND ----------

# MAGIC %md ### Includes

# COMMAND ----------

# MAGIC %run ./python/UtilsTrain

# COMMAND ----------

# MAGIC %run ./python/UtilsPredict

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# Lets tag the automated experiments with "run_"

def create_exp_name(name):
    return create_experiment_name(name,experiment_prefix)

# COMMAND ----------

import time
start_time = time.time()
run_list = []

# COMMAND ----------

def run_train(notebook, exp_name, params, widget_experiment=WIDGET_TRAIN_EXPERIMENT):
    _start_time = time.time()
    exp_name = create_exp_name(exp_name)
    params[widget_experiment] = exp_name
    print("Notebook:",notebook)
    print("  Experiment:",exp_name)
    print("  Params:",params)
    result = dbutils.notebook.run(notebook, 600, params)
    print("  Result:",result)
    toks = result.split(" ")
    run_id, exp_id = toks[0],toks[1]
    print("  Run ID:",run_id)
    run_uri = create_run_uri(exp_id, run_id)
    print("  Run URI:",run_uri)
    _duration = time.time() - _start_time
    print("  Duration:",round(_duration,2))
    run_list.append((exp_name,run_uri,notebook,_duration))
    return run_id

# COMMAND ----------

# MAGIC %run ../tools/dump_run.py

# COMMAND ----------

RUN_MODE = " Run Mode" # Note: leading space to make this widget appear left-most
RUN_ID = "Run ID" 

def run_predict(notebook, run_id):
    print("Notebook:",notebook)
    print("  Run ID:",run_id)
    dbutils.notebook.run(notebook, 60, {RUN_MODE: RUN_ID, RUN_ID: run_id} ) 
    dump_run_id(run_id,3)

# COMMAND ----------

# MAGIC %md ## Run Models - Train and Predict

# COMMAND ----------

# MAGIC %md ### Sklearn Model

# COMMAND ----------

run_id = run_train("python/02a_Sklearn_Train", "sklearn", {"maxDepth": 2, "maxLeafNodes": 32})

# COMMAND ----------

run_predict("python/02b_Sklearn_Predict",run_id)

# COMMAND ----------

# MAGIC %md #### Sklearn Model - Train for multiple parameters

# COMMAND ----------

if train_many:
    params = ([  
        (1,32),  
        (2,32),   
        (4,32),   
        (16,32) ])
    for p in params:
        run_train("python/02a_Sklearn_Train", "sklearn", {"maxDepth": p[0], "maxBins": p[1] })

# COMMAND ----------

# MAGIC %md ### PySpark Model

# COMMAND ----------

run_id = run_train("python/03a_PySpark_Train", "pyspark", {"maxDepth": 2, "maxBins": 32})

# COMMAND ----------

run_predict("python/03b_PySpark_Predict",run_id)

# COMMAND ----------

if train_many:
    params = ([  
        (2,64),   
        (0,32) ])
    for p in params:
        run_train("python/03a_PySpark_Train", "pyspark", {"maxDepth": p[0], "maxBins": p[1]})

# COMMAND ----------

# MAGIC %md ### PySpark Model - MLlib+MLflow Integration

# COMMAND ----------

run_id = run_train("python/04a_PySpark_Train_trackMLlib", "pyspark_trackMLlib", {"maxDepth": "0 2", "maxBins": "4 8 16 32"})

# COMMAND ----------

run_predict("python/03b_PySpark_Predict",run_id)

# COMMAND ----------

# MAGIC %md ### Keras/TensorFlow Model

# COMMAND ----------

run_id = run_train("python/05a_Keras_TF_Train", "keras", {"Epochs": 10, "Optimizer": "adam"})

# COMMAND ----------

if not skipped_failed_tests:
    run_predict("python/05b_Keras_TF_Predict",run_id)

# COMMAND ----------

# MAGIC %md ### Scala Spark ML Models

# COMMAND ----------

# MAGIC %md #### MLflow Classic

# COMMAND ----------

run_id = run_train("scala/01_Scala_Train_Classic", "scala", {"maxDepth": 2, "maxBins": 32}, "Experiment")

# COMMAND ----------

run_predict("scala/03_Scala_Predict",run_id)

# COMMAND ----------

if train_many:
    params = ([  
        (1,32),   
        (16,32) ])
    for p in params:
        run_train("scala/01_Scala_Train_Classic", "scala", {"maxDepth": p[0], "maxBins": p[1]},"Experiment")

# COMMAND ----------

# MAGIC %md #### MLflow Context

# COMMAND ----------

run_id = run_train("scala/02_Scala_Train_Context", "scala", {"maxDepth": 2, "maxBins": 32}, "Experiment")

# COMMAND ----------

run_predict("scala/03_Scala_Predict",run_id)

# COMMAND ----------

# MAGIC %md ### Results

# COMMAND ----------

end_time = time.time()
duration = end_time - start_time
duration = int(duration+.5)
fmt = "%Y-%m-%d %H:%M:%S"
print("Start time:",time.strftime(fmt,time.localtime(start_time)))
print("End time:  ",time.strftime(fmt,time.localtime(end_time)))
print("Duration:  ",duration,"seconds")

# COMMAND ----------

buf = "<table cellpadding=5 cellspacing=0 border=1 width=100%>\n"
buf += "<td>Model</td><td>Seconds</td><td>Experiment Path</td><td>Experiment Link</td>\n"
for tup in run_list:
    exp_name,uri,notebook,duration = tup
    duration = int(round(duration,0))
    buf += "<tr>"
    buf += '<td>{}</td><td>{}</td><td>{}</td><td><a href="{}">{}</a></td>'.format(notebook,duration,exp_name,uri,uri)
    buf += "</tr>\n"
buf += "</table>"
displayHTML(buf)

# COMMAND ----------

run_manager.list_runs()
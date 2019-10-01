# Databricks notebook source
# MAGIC %md ### Find Best Run of an experiment
# MAGIC 
# MAGIC * Finds the best run of an experiment by searching for the min or max of a metric.
# MAGIC * Uses: [Best_Run_Utils](https://demo.cloud.databricks.com/#notebook/2960189)
# MAGIC 
# MAGIC Samples:
# MAGIC * 2100777 - auroch - min
# MAGIC * 2386160 - max_bins

# COMMAND ----------

# MAGIC %md #### Setup

# COMMAND ----------

# MAGIC %run ../common/Best_Run_Utils

# COMMAND ----------

default_experiment_id = "2630330" # /Shared/experiments/demo/sk_WineQuality_demo
default_metric = "rmse"

# COMMAND ----------

dbutils.widgets.text("Experiment ID", default_experiment_id)
dbutils.widgets.text("Metric", default_metric)
dbutils.widgets.dropdown("isLargerBetter","no",["yes","no"])

experiment_id = int(dbutils.widgets.get("Experiment ID"))
metric = dbutils.widgets.get("Metric")
isLargerBetter = dbutils.widgets.get("isLargerBetter") == "yes"

experiment_id,metric,isLargerBetter

# COMMAND ----------

# MAGIC %md #### Calls to find best run

# COMMAND ----------

get_best_run(experiment_id,metric,isLargerBetter)

# COMMAND ----------

# MAGIC %md #### Call to find last run

# COMMAND ----------

get_last_run(experiment_id)
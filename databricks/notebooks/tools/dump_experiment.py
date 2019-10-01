# Databricks notebook source
# MAGIC %md ### Dump experiment with all its run and artifact details
# MAGIC * Shows params, metrics and tags
# MAGIC * Recursively shows all artifacts up to the specified level
# MAGIC * Note: Makes lots of calls to API, so beware running with Default experiment 0

# COMMAND ----------

dbutils.widgets.text(" Experiment ID or name", "1")
dbutils.widgets.text("Artifact Max Level", "1")
dbutils.widgets.dropdown("Show runs","yes",["yes","no"])

experiment_id_or_name = dbutils.widgets.get(" Experiment ID or name")
artifact_max_level = int(dbutils.widgets.get("Artifact Max Level"))
show_runs = dbutils.widgets.get("Show runs") == "yes"

experiment_id_or_name, artifact_max_level, show_runs

# COMMAND ----------

# MAGIC %run ./dump_experiment.py

# COMMAND ----------

dump_experiment(experiment_id_or_name, artifact_max_level, show_runs)
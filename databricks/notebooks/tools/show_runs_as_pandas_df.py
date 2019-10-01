# Databricks notebook source
# MAGIC %md ## Shows runs as Pandas dataframe
# MAGIC 
# MAGIC Overview
# MAGIC * Displays all details of an experiment's runs as a Pandas dataframe.
# MAGIC * Flattens all runs and creates a sparse Pandas dataframe.
# MAGIC * Sparse is needed as runs do not have to have the same metrics, params or tags.
# MAGIC * For each run, all fields of info, data.params, data.metrics and data.tags are flattened into one sparse dict.
# MAGIC * To prevent name clashes in the flattened space, parameters are prefixed with `_p_`, metrics with `_m_` and tags with `_t_`.
# MAGIC *   For example parameter `alpha` becomes `_p_alpha`.
# MAGIC 
# MAGIC Widgets
# MAGIC * CSV file - Write dataframe as CSV if specified
# MAGIC * Sort columns - Opinionated sort of columns by info, params, metrics and tags
# MAGIC * Format time - Format long time as in `2019-07-02 22:12:04`
# MAGIC * Show duration - end_time - start_time
# MAGIC * Faster API calls - use search_runs() instead of iterating over get_run(). 
# MAGIC   * Note search_runs maxes out at 1000 runs. 
# MAGIC   * get_run approach returns all runs but will be much slower when experiment has many runs.

# COMMAND ----------

dbutils.widgets.text(" Experiment ID or name", "")
dbutils.widgets.dropdown("Sort columns","yes",["yes","no"])
dbutils.widgets.dropdown("Format time","yes",["yes","no"])
dbutils.widgets.dropdown("Show duration","yes",["yes","no"])
dbutils.widgets.dropdown("Faster API calls","no",["yes","no"])
dbutils.widgets.dropdown("Skip params","no",["yes","no"])
dbutils.widgets.dropdown("Skip metrics","no",["yes","no"])
dbutils.widgets.dropdown("Skip tags","no",["yes","no"])
dbutils.widgets.text("CSV file","")


exp_id_or_name = dbutils.widgets.get(" Experiment ID or name")
sort_columns = dbutils.widgets.get("Sort columns") == "yes"
format_time = dbutils.widgets.get("Format time") == "yes"
show_duration = dbutils.widgets.get("Show duration") == "yes"
faster_api_calls = dbutils.widgets.get("Faster API calls") == "yes"
skip_params = dbutils.widgets.get("Skip params") == "yes"
skip_metrics = dbutils.widgets.get("Skip metrics") == "yes"
skip_tags = dbutils.widgets.get("Skip tags") == "yes"
csv_file = dbutils.widgets.get("CSV file")

exp_id_or_name, sort_columns, format_time, show_duration, faster_api_calls, skip_params, skip_metrics, skip_tags

# COMMAND ----------

# MAGIC %run ./runs_to_pandas_converter.py

# COMMAND ----------

# MAGIC %run ./mlflow_utils

# COMMAND ----------

import mlflow
client = mlflow.tracking.MlflowClient()
exp = get_experiment(client, exp_id_or_name)
exp_id = exp.experiment_id

# COMMAND ----------

if faster_api_calls:
    runs = client.search_runs([exp_id],"")
else:
    runs = [ client.get_run(info.run_id) for info in client.list_run_infos(exp_id) ]

# COMMAND ----------

converter = RunsToPandasConverter(do_sort=sort_columns, do_pretty_time=format_time, do_duration=True, \
                                 skip_params=skip_params, skip_metrics=skip_metrics, skip_tags=skip_tags)
df = converter.to_pandas_df(runs)
display(df)

# COMMAND ----------

 if csv_file:
    print("CSV file:",csv_file)
    with open(csv_file, 'w') as f:
        df.to_csv(f, index=False)
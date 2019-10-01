# Databricks notebook source
# MAGIC %md ### Dump run details
# MAGIC * Shows params, metrics and tags
# MAGIC * Recursively shows all artifacts up to the specified level
# MAGIC * Note: Makes lots of calls to API, so beware of experiments with many runs

# COMMAND ----------

dbutils.widgets.text(" Run ID", "")
run_id = dbutils.widgets.get(" Run ID")

dbutils.widgets.text("Artifact Max Level", "1")
artifact_max_level = int(dbutils.widgets.get("Artifact Max Level"))

# COMMAND ----------

# MAGIC %run ./dump_run.py

# COMMAND ----------

run = dump_run_id(run_id, artifact_max_level)

# COMMAND ----------

host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
uri = "https://{}/#mlflow/experiments/{}/runs/{}".format(host_name,run.info.experiment_id,run_id)
displayHTML("""<b>Run URI:</b> <a href="{}">{}</a>""".format(uri,uri))
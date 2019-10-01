# Databricks notebook source
# MAGIC %md ### Tests MlflowHttpClient  
# MAGIC * See notebook [mlflow_http_client](https://demo.cloud.databricks.com/#notebook/3652178)
# MAGIC * See: https://mlflow.org/docs/latest/rest-api.html

# COMMAND ----------

# MAGIC %run ./mlflow_http_client

# COMMAND ----------

import json
client = MlflowHttpClient()

# COMMAND ----------

# MAGIC %md #### Test GET

# COMMAND ----------

rsp = client.get("experiments/list")
rsp = json.loads(rsp)
rsp

# COMMAND ----------

experiments = rsp['experiments']
for exp in experiments:
    print(exp)

# COMMAND ----------

# MAGIC %md #### Test POST

# COMMAND ----------

data = { "experiment_ids": [ experiments[0]["experiment_id"]], "query": ""}
rsp = client.post("runs/search", json.dumps(data))
rsp = json.loads(rsp)
rsp

# COMMAND ----------

# MAGIC %md #### Test 404

# COMMAND ----------

client.get("non_existent")
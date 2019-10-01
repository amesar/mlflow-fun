# Databricks notebook source
# MAGIC %md ### MlflowHttpClient - Requests Client for MLflow REST API
# MAGIC * See: https://mlflow.org/docs/latest/rest-api.html
# MAGIC * See notebook [test_mlflow_http_client](https://demo.cloud.databricks.com/#notebook/3652184) for usage

# COMMAND ----------

import requests

class MlflowHttpClient(object):
    def __init__(self):
        self.token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
        self.base_uri = "https://{}/api/2.0/preview/mlflow".format(host_name)
  
    def get(self, path):
        uri = self.create_uri(path)
        rsp = requests.get(uri, headers={'Authorization': 'Bearer '+self.token})
        self.check_response(rsp, uri)
        return rsp.text

    def post(self, path,data):
        uri = self.create_uri(path)
        rsp = requests.post(self.create_uri(path), headers={'Authorization': 'Bearer '+self.token}, data=data)
        self.check_response(rsp, uri)
        return rsp.text
  
    def create_uri(self, path):
        return "{}/{}".format(self.base_uri,path)
    
    def check_response(self, rsp, uri):
        if rsp.status_code < 200 or rsp.status_code > 299:
            raise Exception("HTTP status code: {} Reason: '{}' URL: {}".format(rsp.status_code,rsp.reason,uri))
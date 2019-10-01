# Databricks notebook source
# MAGIC %md Auxiliary run management code for Run_Notebooks

# COMMAND ----------

import mlflow

class RunManager(object):
  def __init__(self, exp_names, base):
    self.client = mlflow.tracking.MlflowClient()
    self.experiment_names = [ base+x for x in exp_names ]

  def delete_runs(self):
    print("Deleting runs:")
    for exp_name in self.experiment_names:
      print("  Experiment: ",exp_name)
      exp = self.client.get_experiment_by_name(exp_name)
      if exp is not None:
        infos = self.client.list_run_infos(exp.experiment_id)
        print("    #runs:",len(infos))
        for info in infos:
          self.client.delete_run(info.run_id)
        infos = self.client.list_run_infos(exp.experiment_id)
      
  def list_runs(self):
    for exp_name in self.experiment_names:
      exp = self.client.get_experiment_by_name(exp_name)
      if exp is None:
        print("?",exp_name)
      else:
        infos = self.client.list_run_infos(exp.experiment_id)
        print(len(infos),exp.experiment_id,exp_name)
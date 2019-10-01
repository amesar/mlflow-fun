# Databricks notebook source
# MAGIC %md dump_run.py
# MAGIC * https://github.com/amesar/mlflow-fun/blob/master/tools/mlflow_fun/tools/dump_run.py
# MAGIC * synchronized 2019-06-28

# COMMAND ----------

"""
Run dump utilities.
"""

import time
import mlflow

INDENT = "  "
MAX_LEVEL = 1
TS_FORMAT = "%Y-%m-%d_%H:%M:%S"
client = mlflow.tracking.MlflowClient()

def dump_run(run, max_level=1, indent=""):
    dump_run_info(run.info,indent)
    print(indent+"Params:")
    for k,v in sorted(run.data.params.items()):
        print(indent+"  {}: {}".format(k,v))
    print(indent+"Metrics:")
    for k,v in sorted(run.data.metrics.items()):
        print(indent+"  {}: {}".format(k,v))
    print(indent+"Tags:")
    for k,v in sorted(run.data.tags.items()):
        print(indent+"  {}: {}".format(k,v))
    print("{}Artifacts:".format(indent))
    dump_artifacts(run.info.run_id, "", 0, max_level, indent+INDENT)
    return run
        
def dump_run_id(run_id, max_level=1, indent=""):
    run = client.get_run(run_id)
    return dump_run(run,max_level,indent)

def dump_run_info(info, indent=""):
    print("{}RunInfo:".format(indent))
    for k,v in sorted(info.__dict__.items()):
        if not k.endswith("_time"):
            print("{}  {}: {}".format(indent,k[1:],v))
    start = _dump_time(info,'_start_time',indent)
    end = _dump_time(info,'_end_time',indent)
    if start is not None and end is not None:
        dur = float(end - start)/1000
        print("{}  _duration:  {} seconds".format(indent,dur))

def _dump_time(info, k, indent=""):
    v = info.__dict__.get(k,None)
    if v is None:
        print("{}  {:<11} {}".format(indent,k[1:]+":",v))
    else:
        stime = time.strftime(TS_FORMAT,time.gmtime(v/1000))
        print("{}  {:<11} {}   {}".format(indent,k[1:]+":",stime,v))
    return v

def dump_artifacts(run_id, path, level, max_level, indent):
    if level+1 > max_level: return
    artifacts = client.list_artifacts(run_id,path)
    for art in artifacts:
        print("{}Artifact - level {}:".format(indent,level))
        for k,v in art.__dict__.items(): print("  {}{}: {}".format(indent,k[1:],v))
        if art.is_dir:
            dump_artifacts(run_id, art.path, level+1, max_level, indent+INDENT)
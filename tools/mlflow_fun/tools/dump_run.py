"""
Run dump utilities.
"""

from __future__ import print_function
import time
from argparse import ArgumentParser
import mlflow

INDENT = "  "
MAX_LEVEL = 1
TS_FORMAT = "%Y-%m-%d_%H:%M:%S"

def dump_run_info(info, name="RunInfo", indent=""):
    print("{}:".format(name))
    for k,v in info.__dict__.items(): 
        if not k.endswith("_time"):
            print("{}  {}: {}".format(indent,k[1:],v))
    start = _dump_time(info,'_start_time',indent)
    end = _dump_time(info,'_end_time',indent)
    if start is not None and end is not None:
        dur = float(end - start)/1000
        print("{}  _duration:  {} seconds".format(indent,dur))

def dump_run(run):
    print("Run")
    dump_run_info(run.info,"  Info","  ")
    print("  Params:")
    for e in run.data.params:
        print("    {}: {}".format(e.key,e.value))
    print("  Metrics:")
    for e in run.data.metrics:
        sdt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(e.timestamp/1000))
        print("    {}: {}  - timestamp: {} {}".format(e.key,e.value,e.timestamp,sdt))
    print("  Tags:")
    for e in run.data.tags:
        print("    {}: {}".format(e.key,e.value))
        
def dump_run_id(run_id, client=None):
    if client is None: client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    dump_run(run)

def _dump_time(info, k, indent=""):
    v = info.__dict__.get(k,None)
    if v is None:
        print("{}  {:<11} {}".format(indent,k[1:]+":",v))
    else:
        stime = time.strftime(TS_FORMAT,time.gmtime(v/1000))
        print("{}  {:<11} {}   {}".format(indent,k[1:]+":",stime,v))
    return v

def dump_artifact(art, indent="", level=0):
    print("{}Artifact - level {}:".format(indent,level))
    for k,v in art.__dict__.items(): print("  {}{}: {}".format(indent,k[1:],v))

def _dump_artifacts(client, run_id, path, indent, level, max_level):
    level += 1
    if level > max_level: return
    artifacts = client.list_artifacts(run_id,path)
    for art in artifacts:
        dump_artifact(art,indent+INDENT,level)
        if art.is_dir:
            _dump_artifacts(client, run_id, art.path, indent+INDENT,level,max_level)

def dump_artifacts(client, run_id, path="", indent="", max_level=MAX_LEVEL):
    print("{}Artifacts:".format(indent))
    _dump_artifacts(client, run_id, path, indent, 0, max_level)


def dump_run_id_with_artifacts(run_id, max_level=1, client=None):
    if client is None: client = mlflow.tracking.MlflowClient()
    dump_run_id(run_id)
    dump_artifacts(client, run_id, path="", indent="  ", max_level=max_level) # XX

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="Run ID", required=True)
    parser.add_argument("--artifact_max_level", dest="artifact_max_level", help="Number of artifact levels to recurse", required=False, default=1, type=int)
    args = parser.parse_args()
    print("args:",args)

    dump_run_id_with_artifacts(args.run_id, args.artifact_max_level)


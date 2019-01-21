
# mlflow-fun/examples/hello_world

Simple Hello World experiment that demonstrates the different ways to run an MLflow experiment.

For details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Synopsis of [hello_world.py](hello_world.py):
* Logs parameters, metrics and tags
* No ML training
* Optionally writes an artifact
* See also [MLproject](MLproject)

The different ways to run an experiment:
* Unmanaged without mlflow
* mlflow local
* mlflow git
* mlflow remote

## Setup

**External tracking server**
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```

**Databricks managed tracking server**
```
export MLFLOW_TRACKING_URI=databricks
```
The token and tracking server URL will be picked up from your Databricks CLI ~/.databrickscfg default profile.

## Running

### Unmanaged without mlflow
```
python hello_world.py .01 without_mlflow True
```

### mlflow local
```
mlflow run . -Palpha=.01 -Prun_origin=LocalRun -Plog_artifact=True
```
You can also specify an experiment ID:
```
mlflow run . --experiment-id=1 -Palpha=.01 -Prun_origin=LocalRun -Plog_artifact=True
```

### mlflow git
```
mlflow run  https://github.com/amesar/mlflow-fun.git#examples/hello_world \
  -Palpha=100 -Prun_origin=GitRun -Plog_artifact=True
```
### mlflow Databricks remote
Run against Databricks. See [Remote Execution on Databricks](https://mlflow.org/docs/latest/projects.html#remote-execution-on-databricks) and [cluster.json](wine-quality/cluster.json).
```
mlflow run  https://github.com/amesar/mlflow-fun.git#examples/hello_world \
  -Palpha=100 -Prun_origin=RemoteRun -Plog_artifact=True \
  -m databricks --cluster-spec cluster.json
```

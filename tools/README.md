# mlflow-fun/tools

## Overview

Useful tools for MLflow.
* Dumps experiment or run information recursively.
* Exports a run from one MLflow server and imports it into another server.
* [MLflow Analytics](mlflow_analytics) - Create Spark tables to query experiment and run data obtained from API
* [MLflow Metrics](mlflow_metrics) - Create Spark tables to query for best metric
* [Mlflow Fun Common](mlflow_fun_common) - Common MLflow Fun stuff

## Dump experiment or run

Dumps all experiment or run information recursively.

```
python dump_run.py --run_id 2cbab69842e4412c99bfb5e15344bc42 --artifact_max_level 5 
  
python dump_experiment.py --experiment_id 123 --showRuns --artifact_max_level 5
```

Example for dump_experiment.py
```
Experiment Details:
  experiment_id: 123
  name: py/sk/ElasticNet/WineQuality
  artifact_location: /Users/andre/work/mlflow/local_mlrun/mlruns/1
  lifecycle_stage: active
  #runs: 3
Run 2cbab69842e4412c99bfb5e15344bc42
  run_uuid: 2cbab69842e4412c99bfb5e15344bc42
  experiment_id: 123
  name:
  source_type: 4
  source_name: train_wine_quality.py
  entry_point_name:
  user_id: andre
  status: 3
  start_time: 1550849061031
  end_time: 1550849061685
  source_version: b84cb9ea744d27632eedcc3aa79974ea3c360927
  lifecycle_stage: active
  artifact_uri: /Users/andre/work/mlflow/local_mlrun/mlruns/1/2cbab69842e4412c99bfb5e15344bc42/artifacts
  Params:
    alpha: 0.5
    l1_ratio: 0.5
  Metrics:
    mae: 0.6278761410160693  - timestamp: 1550849061 1970-01-18 22:47:29
    r2: 0.12678721972772677  - timestamp: 1550849061 1970-01-18 22:47:29
    rmse: 0.82224284975954  - timestamp: 1550849061 1970-01-18 22:47:29
  Tags:
    platform: Darwin
  Artifacts:
    Artifact - level 1:
      path: model
      is_dir: True
      bytes: None
      Artifact - level 2:
        path: model/MLmodel
        is_dir: False
        bytes: 343
. . .
```

## Export/Import Run

Exports a run from one MLflow server and imports it into another server.

In this example use:

* Experiment [../examples/scikit-learn/wine-quality](../examples/scikit-learn/wine-quality)
* Source tracking server runs on port 5000 
* Destination server on 5001

**Create new destination experiment**

First create an experiment in the destination MLflow tracking server.
```
export MLFLOW_TRACKING_URI=http://localhost:5001
mlflow experiments create new_experiment

Created experiment 'local_runs' with id 1
```

**Export and import the run**

Then run [export_import_run.py](export_import_run.py). 

```
export MLFLOW_TRACKING_URI=http://localhost:5000

python export_import_run.py \
  --src_run_id 50fa90e751eb4b3f9ba9cef0efe8ea30 \
  --dst_experiment_id 123 \
  --dst_uri http://localhost:5001
  --log_source_info
```

The log_source_info creates metadata tags with import information. Sample tags from UI:
```
Name                         Value
_exim_import_timestamp       1551037752
_exim_import_timestamp_nice  2019-02-24 19:49:12
_exim_src_experiment_id      2368826
_exim_src_run_id             d9bbe1c8c94f434b8514301a0834063c
_exim_src_uri                http://localhost:5000
```

**Check predictions from new run**

Check the predictions from the new run in the destination server.

```
export MLFLOW_TRACKING_URI=http://localhost:5001
cd ../examples/scikit-learn/wine-quality
python scikit_predict.py bf3890a927fb4c82be37221eed8069d7

data_path: wine-quality.csv
run_id: bf3890a927fb4c82be37221eed8069d7
model: ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=2.0,
      max_iter=1000, normalize=False, positive=False, precompute=False,
      random_state=42, selection='cyclic', tol=0.0001, warm_start=False)
predictions: [5.55576406 5.54277365 5.89895397 ... 5.73139958 6.23485413 6.12031375]
```

# mlflow-fun tools 

## Overview

Some useful tools for MLflow.
* Dumps experiment or run information recursively.


## Dump experiment or run

Dumps all experiment or run information recursively.

```
export PYTHONPATH=../..

python dump_run.py --run_id 2cbab69842e4412c99bfb5e15344bc42 --artifact_max_level 5 
  
python dump_experiment.py --experiment_id 2 --show_runs --artifact_max_level 5
```

Example for [dump_experiment.py](dump_experiment.py)
```
Experiment Details:
  experiment_id: 3
  name: sklearn_wine_elasticNet
  artifact_location: /Users/andre/work/mlflow/local_mlrun/mlruns/3
  lifecycle_stage: active
  #runs: 2
Run
  Info:
    run_uuid: fdc33f23d2ac4b0bae5f8181700c00ed
    experiment_id: 3
    name: 
    source_type: 4
    source_name: train.py
    entry_point_name: 
    user_id: andre
    status: 3
    source_version: 47e8ec307671203cf5607ac2534cbd8fe5e05677
    lifecycle_stage: active
    artifact_uri: /Users/andre/work/mlflow/local_mlrun/mlruns/3/fdc33f23d2ac4b0bae5f8181700c00ed/artifacts
    start_time: 2019-06-04_19:59:30   1559678370412
    end_time:   2019-06-04_19:59:32   1559678372819
    _duration:  2.407 seconds
  Params:
    l1_ratio: 0.5
    alpha: 0.001
  Metrics:
    mae: 0.5837497243928481  - timestamp: 1559678372 1970-01-19 01:14:38
    r2: 0.2726475054853086  - timestamp: 1559678372 1970-01-19 01:14:38
    rmse: 0.7504340478812797  - timestamp: 1559678372 1970-01-19 01:14:38
  Tags:
    data_path: ../../data/wine-quality/wine-quality-white.csv
    mlflow.source.git.commit: 47e8ec307671203cf5607ac2534cbd8fe5e05677
    platform: Darwin
    mlflow.source.name: train.py
    mlflow.source.type: LOCAL
  Artifacts:
    Artifact - level 1:
      path: sklearn-model
      is_dir: True
      bytes: None
      Artifact - level 2:
        path: sklearn-model/MLmodel
        is_dir: False
        bytes: 351
      Artifact - level 2:
        path: sklearn-model/conda.yaml
        is_dir: False
        bytes: 119
      Artifact - level 2:
        path: sklearn-model/model.pkl
        is_dir: False
        bytes: 627
    Artifact - level 1:
      path: wine_ElasticNet-paths.png
      is_dir: False
      bytes: 27772
```

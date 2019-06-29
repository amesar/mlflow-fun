# mlflow-fun tools 

## Overview

Some useful tools for MLflow.

## Dump experiment or run
Dumps all experiment or run information recursively.

**Overview**
* [dump_run.py](dump_run.py) - Dumps run information.
  * Shows info, params, metrics and tags.
  * Recursively shows all artifacts up to the specified level.
* [dump_experiment.py](dump_experiment.py) - Dumps experiment information.
  * If `show_runs` is true, beware of experiments with many runs as each run results in an API call.
* A large value for `artifact_max_level` also incurs many API calls.


**Run dump tools**
```
export PYTHONPATH=../..

python dump_run.py --run_id 2cbab69842e4412c99bfb5e15344bc42 --artifact_max_level 5 
  
python dump_experiment.py --experiment_id 1812 --show_runs --artifact_max_level 5
```

**Sample output for dump_experiment.py**
```
Experiment Details:
  experiment_id: 1812
  name: sklearn_wine_elasticNet
  artifact_location: /opt/mlflow/server/mlruns/1812
  lifecycle_stage: active
  #runs: 5
Runs:
  Run 0:
    RunInfo:
      run_uuid: fdc33f23d2ac4b0bae5f8181700c00ed
      experiment_id: 1812
      name: 
      source_type: 4
      source_name: train.py
      entry_point_name: 
      user_id: andre
      status: 3
      source_version: 47e8ec307671203cf5607ac2534cbd8fe5e05677
      lifecycle_stage: active
      artifact_uri: /opt/mlflow/server/mlruns/1812/fdc33f23d2ac4b0bae5f8181700c00ed/artifacts
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

# mlflow-fun/tools - Core

## Overview

Some useful tools for MLflow.
* Exports a run from one MLflow server and imports it into another server.

## Export/Import Run

Exports a run from one MLflow server and imports it into another server.

In this example we use:

* Experiment [../../examples/scikit-learn/wine-quality](../examples/scikit-learn/wine-quality)
* Source tracking server runs on port 5000 
* Destination server runs on 5001

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

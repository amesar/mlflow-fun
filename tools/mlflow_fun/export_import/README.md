# mlflow-fun/tools - Core

## Export/Import Run

Exports a run from one MLflow tracking server and imports it into another server.

In this example we use:

* Experiment [../../examples/sklearn/wine-quality](../../examples/sklearn/wine-quality)
* Source tracking server runs on port 5000 
* Destination tracking server runs on 5001

**Export and import the run**

Run [export_import_run.py](export_import_run.py). 

```
export MLFLOW_TRACKING_URI=http://localhost:5000

python export_import_run.py \
  --src_run_id 50fa90e751eb4b3f9ba9cef0efe8ea30 \
  --dst_experiment_id_name my_experiment \
  --dst_uri http://localhost:5001
  --log_source_info
```

The `log_source_info` option creates metadata tags with import information. Sample tags from UI:
```
Name                         Value
_exim_import_timestamp       1551037752
_exim_import_timestamp_nice  2019-02-24 19:49:12
_exim_src_experiment_id      2368826
_exim_src_run_id             50fa90e751eb4b3f9ba9cef0efe8ea30
_exim_src_uri                http://localhost:5000
```

**Check predictions from new run**

Check the predictions from the new run in the destination server.

```
export MLFLOW_TRACKING_URI=http://localhost:5001
cd ../../examples/sklearn/wine-quality
python scikit_predict.py bf3890a927fb4c82be37221eed8069d7
```

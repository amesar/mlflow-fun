# mlflow-fun/tools - Export/Import Experiments or Runs

Tools to export and import MLflow experiments or runs.

## Overview

* Experiments
  * Export an experiment and all its runs to a directory or zip file
  * Import an experiment from directory or zip file
  * Copy an experiment from one tracking server to another
* Runs
  * Export a run to directory or zip file
  * Import a run from directory or zip file
  * Copy a run from one tracking server to another

* Copy Notes
  * Copy logic is still under development
  * It works pretty well with open-source MLflow
  * Does not yet work for Databricks MLflow

* TODO
  * Account for nested runs
  * Implement Databricks notebook imports

### Common arguments 

`output` - Can either be a directory or zip file (`output` has a zip extension).

`intput` - Can either be a directory or zip file (if `output` has a zip extension).

`notebook_formats` - If exporting a Databricks experiment, the run's notebook can be saved in the specified formats (comma-delimited argument). Each format is saved as `notebook.{format}`. Supported formats are  SOURCE, HTML, JUPYTER, DBC. See [Export Format](https://docs.databricks.com/dev-tools/api/latest/workspace.html#notebookexportformat) documentation.

`log_source_info` - Creates metadata tags (starting with `mlflow_tools.export`) containing export information.
```
Name                                Value
mlflow_tools.export.timestamp       1551037752
mlflow_tools.export.timestamp_nice  2019-02-24 19:49:12
mlflow_tools.export.experiment_id   2
mlflow_tools.export.experiment_name sklearn_wine
mlflow_tools.export.run_id          50fa90e751eb4b3f9ba9cef0efe8ea30
mlflow_tools.export.tracking_uri    http://localhost:5000
```

## Experiments

### Export experiment

Export an experiment to a directory or zip file.

Arguments
* experiment - Source experiment name or ID
* output - Destination directory or zip file

#### Export example
```
python export_experiment.py --experiment=2 --output=out --log_source_info
```
```
python export_experiment.py --experiment=sklearn_wine --output=exp.zip
```

#### Databricks export example
See the [root REAMDE.md Databricks section](../../../README.md#Databricks) for detailed Databricks configuration information.
```
export MLFLOW_TRACKING_URI=databricks
export DATABRICKS_HOST=https://acme.cloud.databricks.com
export DATABRICKS_TOKEN=MY_TOKEN

python export_experiment.py --experiment=sklearn_wine --notebook_formats=DBC,SOURCE
```

#### Output 

Output export directory example
```
manifest.json
130bca8d75e54febb2bfa46875a03d59/
5a22839d66154001882e0632581fbf02/
```

manifest.json - source experiment information
```
{
  "experiment": {
    "experiment_id": "2",
    "name": "sklearn_wine",
    "artifact_location": "/opt/mlflow/server/mlruns/2",
    "lifecycle_stage": "active"
  },
  "export_info": {
    "export_time": "2019-07-21 13:36:28",
    "num_runs": 2
  },
  "run_ids": [
    "130bca8d75e54febb2bfa46875a03d59",
    "5a22839d66154001882e0632581fbf02"
  ],
  "failed_run_ids": []
}
```

### Import experiment

Import an experiment from a directory or zip file.

Arguments
* experiment_name - Destination experiment name  - will be created if it does not exist
* input - Source directory or zip file produced by export_experiment.py

Run examples

```
python import_experiment.py \
  --experiment_name=sklearn_wine \
  --input=out 
```
```
python import_experiment.py \
  --experiment_name=sklearn_wine \
  --input=exp.zip 
```

### Copy experiment from one tracking server to another

Copies an experiment from one MLflow tracking server to another.

Source: [copy_experiment.py](copy_experiment.py)

In this example we use
* Source tracking server runs on port 5000 
* Destination tracking server runs on 5001

Arguments
* src_experiment - Source experiment name or ID
* dst_experiment_name - Destination experiment name  - will be created if it does not exist
* src_uri - Source server URI
* dst_uri - Destination server URI

Run example
```
export MLFLOW_TRACKING_URI=http://localhost:5000

python copy_experiment.py \
  --src_experiment=sklearn_wine \
  --dst_experiment_name sklearn_wine_imported \
  --src_uri http://localhost:5000
  --dst_uri http://localhost:5001
  --log_source_info
```

## Runs

### Export run

Export run to directory or zip file.

Arguments
* run_id - Source run ID
* output - Destination directory or zip file

Run examples
```
python export_run.py \
  --run_id=50fa90e751eb4b3f9ba9cef0efe8ea30 \
  --output=out
  --log_source_info
```
```
python export_run.py \
  --run_id=50fa90e751eb4b3f9ba9cef0efe8ea30 \
  --output=run.zip
```

Produces a directory with the following structure:
```
run.json
artifacts
  plot.png
  sklearn-model
    MLmodel
    conda.yaml
    model.pkl
```
Sample run.json
```
{   
  "info": {
    "run_id": "130bca8d75e54febb2bfa46875a03d59",
    "experiment_id": "2",
    ...
  },
  "params": {
    "max_depth": "16",
    "max_leaf_nodes": "32"
  },
  "metrics": {
    "mae": 0.5845562996214364,
    "r2": 0.28719674214710467,
  },
  "tags": {
    "mlflow.source.git.commit": "a42b9682074f4f07f1cb2cf26afedee96f357f83",
    "mlflow.runName": "demo.sh",
    "run_origin": "demo.sh",
    "mlflow.source.type": "LOCAL",
    "mlflow_tools.export.tracking_uri": "http://localhost:5000",
    "mlflow_tools.export.timestamp": 1563572639,
    "mlflow_tools.export.timestamp_nice": "2019-07-19 21:43:59",
    "mlflow_tools.export.run_id": "130bca8d75e54febb2bfa46875a03d59",
    "mlflow_tools.export.experiment_id": "2",
    "mlflow_tools.export.experiment_name": "sklearn_wine"
  }
}
```

### Import run

Imports a run from a directory or zip file.

Arguments
* experiment_name - Destination experiment name  - will be created if it does not exist
* input - Source directory or zip file produced by export_run.py

Run examples
```
python import_run.py \
  --run_id=50fa90e751eb4b3f9ba9cef0efe8ea30 \
  --input=out \
  --experiment_name=sklearn_wine2
```

### Copy run from one tracking server to another

Copies a run from one MLflow tracking server to another.

Source: [copy_run.py](copy_run.py)

In this example we use
* Source tracking server runs on port 5000 
* Destination tracking server runs on 5001

Arguments
* src_run_id - Source run ID
* dst_experiment_name - Destination experiment name  - will be created if it does not exist
* src_uri - Source server URI
* dst_uri - Destination server URI

Run example
```
export MLFLOW_TRACKING_URI=http://localhost:5000

python copy_run.py \
  --src_run_id=50fa90e751eb4b3f9ba9cef0efe8ea30 \
  --dst_experiment_name sklearn_wine \
  --src_uri http://localhost:5000
  --dst_uri http://localhost:5001
  --log_source_info
```

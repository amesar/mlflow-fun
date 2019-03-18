# mlflow-fun - common

Common utilities for MLflow Fun repo.

## Overview

Files:
* [http_client.py](mlflow_fun/common/http_client.py) -  HttpClient - Wrapper for get and post methods for MLflow REST API
* [mlflow_search_client.py](mlflow_fun/common/mlflow_search_client.py) - MlflowSearchClient - Client with extra functionality based on REST endpoint runs/search
* [mlflow_utils.py](mlflow_fun/common/mlflow_utils.py) - MLflow utilities
* [databricks_cli_utils.py](mlflow_fun/common/databricks_cli_utils.py) - Databricks CLI utilities

MLflow tracking URI and token are picked up per standard MLflow environment variables.

## Sample Usage

### HttpClient
```
from mlflow_fun.common.http_client import HttpClient
client = HttpClient()
rsp = client.get("experiments/list")
for exp in rsp['experiments']:
     print("  ",exp)
```

### MlflowSearchClient
```
from mlflow_fun.common.mlflow_search_client import MlflowSearchClient
search_client = MlflowSearchClient()
runs = search_client.list_runs_flat(2019)
for run in runs:
     print("  ",run)
```

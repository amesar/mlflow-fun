# mlflow-fun - common

Common utilities for MLflow Fun repo.

## Overview

Files:
* [http_client.py](http_client.py) -  HttpClient - Wrapper for get and post methods for MLflow REST API
* [mlflow_utils.py](mlflow_utils.py) - MLflow utilities
* [databricks_cli_utils.py](databricks_cli_utils.py) - Databricks CLI utilities
* [search_runs_iterator.py](search_runs_iterator.py) - Convenience tterator for search_runs() that takes care of paged_list.token.

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

### SearchRunsIterator
```
from mlflow_fun.common.search_runs_iterator import SearchRunsIterator
it = SearchRunsIterator(mlflow_client, [experiment_id], MAX_RESULTS)
for run in it:
     print("  ",run)
```

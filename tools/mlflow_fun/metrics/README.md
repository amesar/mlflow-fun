# mlflow-fun/tools - MLflow Metrics

## Overview

Get the best run for an experiment by querying its run metrics.

Several ways to query an experiment's run metric data:
* Directly call API and manipulate response
* Use Pandas Dataframe API
* Use Spark Dataframe API or SQL

Notes:
* Sample is based on the [Wine Quality](../../examples/scikit-learn/wine-quality) experiment.
* Parameter columns are prefixed with `_p_` and metrics start with `_m_`.
* Data is obtained by calling the MLflow API - either Python or REST API.

## Pandas Best Run

Creates a dataframe for an experiment with each row containing the run metadata, parameters and metrics.

Files:
  * [main_dataframe_builder.py](pandas/main_dataframe_builder.py) - Sample for PandasDataframeBuilder
  * [dataframe_builder.py](pandas/dataframe_builder.py) - PandasDataframeBuilder code

Find best run for a metric.
```
from mlflow_fun.metrics.pandas.dataframe_builder import PandasDataframeBuilder
builder = PandasDataframeBuilder()
best = builder.get_best_run(experiment_id,"_m_rmse",ascending)
print("best:",best)
```

```
best: ('0e66276c6fa4489aa55dc5bc80c58711', 0.7497487101907394)
```

Code to find best run for a metric.
```
metric = "_m_rmse"
df = builder.build_dataframe(experiment_id)
df = df[['run_uuid',metric]]
df = df.sort_values(metric,ascending=ascending)
best = df.iloc[0]
best = (best[0],best[1])
print("best:",best)
```
```
best: ('0e66276c6fa4489aa55dc5bc80c58711', 0.7497487101907394)
```

## Spark Best Run

Creates a dataframe or table of runs for an experiment with each row containing the run metadata, parameters and metrics.

Files:
  * [main_build_tables.py](spark/main_build_tables.py) - Builds tables for experiments.
  * [table_builder.py](spark/table_builder.py) - Builds tables core.
  * [dataframe_builder.py](spark/dataframe_builder.py) - Builds a dataframe.

### Dataframe Usage
```
from mlflow_fun.metrics.spark.dataframe_builder import FastDataframeBuilder
from pyspark.sql.functions import round

builder = FastDataframeBuilder()
df = builder.build_dataframe(experiment_id)
df = df.select("run_uuid", round("_m_rmse",2).alias("_m_rmse"), "_p_alpha", "_p_l1_ratio").sort("_m_rmse")
```

```
+--------------------------------+-------+--------+-----------+
|run_uuid                        |_m_rmse|_p_alpha|_p_l1_ratio|
+--------------------------------+-------+--------+-----------+
|61fb99c76031475e8c7ca11f147672f0|0.82   |0.5     |0.5        |
|2c535a4484fe4852af85292d637838de|0.82   |0.5     |0.5        |
|fae7c2dedb74489fbceb95f228346c35|0.82   |0.5     |0.5        |
|6ebeca3a2e8b42419b6b4c701834c096|0.86   |2.0     |0.5        |
|c38211cc0e624767b5885ece179689fe|0.87   |9.0     |0.5        |
|ff8c4758bb3f4be1af45eb22bd615791|0.87   |9.0     |0.5        |
|ceb6226354234d568b4d8d5a52130962|0.87   |3.0     |0.5        |
+--------------------------------+-------+--------+-----------+
```

### SQL Table Usage

#### Build tables

##### Build tables for experiments

If `--experiment_ids` is not specified, then all experiments will be processed.
```
spark-submit --master local[2] main_build_tables.py \
   --database mlflow_metrics \
   --data_dir /opt/mlflow/databases/mlflow_metrics \
   --experiment_ids 1,2,9
```

#### Queries

```
describe table exp_1

+----------------+---------+-------+
|        col_name|data_type|comment|
+----------------+---------+-------+
|          _m_mae|   double|   null|
|           _m_r2|   double|   null|
|         _m_rmse|   double|   null|
|        _p_alpha|   double|   null|
|    _p_data_path|   string|   null|
|       _p_exp_id|      int|   null|
|     _p_exp_name|   string|   null|
|     _p_l1_ratio|   double|   null|
|   _p_run_origin|   string|   null|
|    artifact_uri|   string|   null|
|        end_time|   bigint|   null|
|entry_point_name|   string|   null|
|   experiment_id|      int|   null|
| lifecycle_stage|   string|   null|
|            name|   string|   null|
|        run_uuid|   string|   null|
|     source_name|   string|   null|
|     source_type|      int|   null|
|  source_version|   string|   null|
|      start_time|   bigint|   null|
+----------------+---------+-------+
```

Find the best `rmse`.
```
select run_uuid, round(_m_rmse,2) as _m_rmse, _p_alpha, _p_l1_ratio from exp_1 order by _m_rmse

+--------------------------------+-------+--------+-----------+
|run_uuid                        |_m_rmse|_p_alpha|_p_l1_ratio|
+--------------------------------+-------+--------+-----------+
|61fb99c76031475e8c7ca11f147672f0|0.82   |0.5     |0.5        |
|2c535a4484fe4852af85292d637838de|0.82   |0.5     |0.5        |
|fae7c2dedb74489fbceb95f228346c35|0.82   |0.5     |0.5        |
|6ebeca3a2e8b42419b6b4c701834c096|0.86   |2.0     |0.5        |
|c38211cc0e624767b5885ece179689fe|0.87   |9.0     |0.5        |
|ceb6226354234d568b4d8d5a52130962|0.87   |3.0     |0.5        |
|ff8c4758bb3f4be1af45eb22bd615791|0.87   |9.0     |0.5        |
+--------------------------------+-------+--------+-----------+
```


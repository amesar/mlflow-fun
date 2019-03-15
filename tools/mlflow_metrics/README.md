# mlflow-fun/tools - MLflow Metrics

## Overview

Get the best run for an experiment by querying its metrics.

Creates a table of runs for an experiment with each row containing the run metadata, parameters and metrics.
Data is obtained by calling the MLflow API.

Parameter columns are prefixed with `_p_` and metrics start correspondingly with `_m_`.

Files:
  * [exp_main.py](exp_main.py) - Build table for one experiment
  * [all_main.py](all_main.py) - Build table some or all experiments
  * [build_tables.py](mlflow_metrics/build_tables.py) - Build table core

## Build tables

### Build table for one experiment
```
spark-submit --master local[2] exp_main.py \
   --database mlflow_metrics \
   --data_dir /opt/mlflow/databases/mlflow_metrics
   --experiment_id 1
```

### Build tables for some/all experiments

If `--experiment_ids` is not specified, then all experiments will be processed.
```
spark-submit --master local[2] all_main.py \
   --database mlflow_metrics \
   --data_dir /opt/mlflow/databases/mlflow_metrics \
   --experiment_ids 1,2,9
```

## Sample Table 

For the [Wine Quality experiment](../../examples/scikit-learn/wine-quality).

### Schema
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

### Query

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


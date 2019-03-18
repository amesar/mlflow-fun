# mlflow-fun/tools - MLflow SQL Analytics

## Overview

Create Spark tables `experiments` and `runs` to allows queries across all 
[Experiment](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Experiment) and 
[RunInfo](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunInfo) data.

Files:
  * [build_tables.py](mlflow_analytics/build_tables.py)
  * [sample_queries.py](mlflow_analytics/sample_queries.py)

## Build tables

```
export PYTHONPATH=.
spark-submit --master local[2] mlflow_analytics/build_tables.py \
   --database mlflow_analytics \
   --data_dir /opt/mlflow/mlflow_analytics
```

## Table Schemas

### experiments
```
describe table experiments
+-----------------+---------+-------+
|col_name         |data_type|comment|
+-----------------+---------+-------+
|experiment_id    |int      |null   |
|name             |string   |null   |
|artifact_location|string   |null   |
|lifecycle_stage  |string   |null   |
+-----------------+---------+-------+
```

### runs
```
describe table runs
+----------------+---------+-------+
|col_name        |data_type|comment|
+----------------+---------+-------+
|run_uuid        |string   |null   |
|experiment_id   |int      |null   |
|name            |string   |null   |
|source_type     |int      |null   |
|source_name     |string   |null   |
|entry_point_name|string   |null   |
|user_id         |string   |null   |
|status          |int      |null   |
|start_time      |bigint   |null   |
|end_time        |bigint   |null   |
|source_version  |string   |null   |
|lifecycle_stage |string   |null   |
|artifact_uri    |string   |null   |
+----------------+---------+-------+
```

## Sample queries

```
select * from mlflow_status
+-------------------+---------------------+
|refreshed_at       |tracking_uri         |
+-------------------+---------------------+
|2019-03-03 21:35:37|http://localhost:5000|
+-------------------+---------------------+

```

```
select * from experiments
+-------------+---------------+---------------------------------------------+------+
|experiment_id|name           |artifact_location                   |lifecycle_stage|
+-------------+---------------+---------------------------------------------+------+
|0            |Default        |/opt/mlflow/tracking_server/mlruns/0|active         |
|1            |BestWineQuality|/opt/mlflow/tracking_server/mlruns/1|active         |
|2            |test           |/opt/mlflow/tracking_server/mlruns/2|active         |
+-------------+---------------+---------------------------------------------+------+

select e.experiment_id,e.name experiment_name,count(r.experiment_id) as num_runs from runs r
  right outer join experiments e on e.experiment_id=r.experiment_id
  group by e.experiment_id, e.name order by num_runs desc

+-------------+---------------+--------+
|experiment_id|experiment_name|num_runs|
+-------------+---------------+--------+
|1            |BestWineQuality|7       |
|2            |test           |5       |
|0            |Default        |0       |
+-------------+---------------+--------+
```

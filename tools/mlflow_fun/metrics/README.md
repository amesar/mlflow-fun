# mlflow-fun/tools - MLflow Metrics

Get the best run for an experiment by querying its run metrics.

## Overview 

Many people ask how can they get the "best run" for an experiment.
This can be as easy as getting the best (minimum or maximum) value for one metric, or a more complex query involving several metrics.

In this project we explore several approaches to obtaining the "best run".

* Directly call the API and manipulate the response with custom code.
* SQL Database. If you are using a database as your backend store, you can directly query the tables.
* Create a flat table containing all run data (info, metrics, parameters and tags). Then use higher-level APIs to query the table.
  * Use Pandas Dataframe API.
  * Use Spark Dataframe API or SQL.

The API approach is the most obvious solution. The MLflow user calls existing Python API methods to list run details, and then with custom Python logic determines the best run. In the example below we find the run with lowest RMSE value.

The most obvious way is to call existing Python API methods to list run details, and then with custom Python logic determine the best run. In the example below we find the run with lowest RMSE value.

There are several problems with this approach. First of all it involves custom code. It would be preferable to have a higher-level abstraction (Pandas or SQL/Spark) to allow you to determine the best run. For each experiment we build a flattend table `exp_{EXPERIMENT_ID}` containing all of the run data.

Notes:
* Sample is based on the [Wine Quality](../../../examples/scikit-learn/wine-quality) experiment.
* Parameter columns are prefixed with `_p_` and metrics start with `_m_`.
* Data is obtained by calling the MLflow API - either Python or REST API.

## SQL Back-end Database Query

If you are using a database as your backend store, you can issue this query to find the best run.

```
select r.run_uuid, m.value from runs r 
join experiments e on r.experiment_id = e.experiment_id 
join metrics m on r.run_uuid = m.run_uuid 
where e.experiment_id=1 and m.key='rmse'
order by m.value
limit 1
```

```
+----------------------------------+----------+
| run_uuid                         | value    |
+----------------------------------+----------+
| 9a59c7712f914889932d7c8ccc72b775 | 0.749749 |
+----------------------------------+----------+
```

Files:
  * [sql_best.py](sql/sql_best.py) - Code for direct SQL call.
  * [main_sql.py](sql/main_sql.py) - Sample main program.

Run:
```
python main_sql.py
  --connection mysql://my_user:my_password@localhost:3306/mlflow
  --experiment_id=1 --metric=rmse 
```
```
('9a59c7712f914889932d7c8ccc72b775', 0.749749)
```

## Direct API Manipulation

Files:
  * [api_best.py](api/api_best.py) - Code for direct API calls.
  * [main_api.py](api/main_api.py) - Sample main program.

Call the standard MLflow Python API to find the best run. Note this can be slow if you have many runs since a REST call is needed for each run.
```
def get_best_run(experiment_id, metric):
    client = mlflow.tracking.MlflowClient()
    best = None
    for info in client.list_run_infos(experiment_id):
        run = client.get_run(info.run_uuid)
        for m in run.data.metrics:
            if m.key == metric:
               if best is None or m.value < best[1]):
                   best = (info.run_uuid,m.value)
    return best
```
It is more efficient to use the REST API [runs/search](https://mlflow.org/docs/latest/rest-api.html#search-runs) endpoint which makes one call to get all the run details for an experiment.
```
def get_best_run_fast(host, token, experiment_id, metric):
    api_path = "api/2.0/preview/mlflow"
    uri = os.path.join(host,api_path,"runs/search")
    req = '{ "experiment_ids": ['+str(experiment_id)+'] }'
    headers = {} if token is None else {'Authorization': 'Bearer '+token}
    rsp = requests.get(uri, data=req, headers=headers)
    best = None
    for run in json.loads(rsp.text)['runs']:
        if 'data' in run and 'metrics' in run['data']:
            mlist = run['data']['metrics']
            mdct = { m['key'] : m['value'] for m in mlist }
            if metric in mdct:
                mval = mdct[metric]
                if best is None or (isinstance(mval,float) and mval < best[1]):
                    best = (run['info']['run_uuid'],mval)
    return best
```

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

Show Dataframe columns.
```
print(df.types)

_m_mae                         float64
_m_r2                          float64
_m_rmse                        float64
_p_alpha                        object
_p_l1_ratio                     object
_t_mlflow.source.git.commit     object
_t_mlflow.source.name           object
_t_mlflow.source.type           object
artifact_uri                    object
end_time                        object
experiment_id                   object
lifecycle_stage                 object
name                            object
run_uuid                        object
source_name                     object
source_type                     object
source_version                  object
start_time                      object
status                          object
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

experiment_id = 1 # for WineQuality experiment
builder = FastDataframeBuilder()
df = builder.build_dataframe(experiment_id)
df = df.select("run_uuid", round("_m_rmse",4).alias("_m_rmse"), "_p_alpha", "_p_l1_ratio").sort("_m_rmse")
```

```
+--------------------------------+-------+--------+-----------+
|run_uuid                        |_m_rmse|_p_alpha|_p_l1_ratio|
+--------------------------------+-------+--------+-----------+
|0e66276c6fa4489aa55dc5bc80c58711|0.7497 |0.0001  |0.001      |
|3edde31d5d6c4994bb6f55a385ecbe0b|0.7504 |0.001   |0.001      |
|ff9243617b2a423faa53b04bb48b0f89|0.7585 |0.01    |0.001      |
|c9aba721e4324bc68980e66260fd6fe6|0.7758 |0.1     |0.001      |
|1d88d9c3f7754cfc9c80b4fe14ffb059|0.7874 |0.5     |0.0001     |
|6a8ed4640640451d82265ce45e3b45b7|0.7875 |0.5     |0.001      |
|0041a8f619434408b2fb8d424292eddc|0.7883 |0.5     |0.01       |
|c5ab9b9e8401491b9acf5a092de29dec|0.7948 |0.5     |0.1        |
|d82c0e7b8c214db3affdc48519a4defd|0.799  |1.0     |0.001      |
|9f101be55ec44116b03ebbcd1abecf47|0.8222 |0.5     |0.5        |
+--------------------------------+-------+--------+-----------+
```

### SQL Table Usage

#### Build tables

##### Build tables for experiments

If `--experiment_ids` is not specified, then all experiments will be processed.
```
spark-submit --master local[2] \
   mlflow_fun/metrics/spark/main_build_tables.py \
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
select run_uuid, round(_m_rmse,4) as _m_rmse, _p_alpha, _p_l1_ratio from exp_1 order by _m_rmse
+--------------------------------+-------+--------+-----------+
|run_uuid                        |_m_rmse|_p_alpha|_p_l1_ratio|
+--------------------------------+-------+--------+-----------+
|0e66276c6fa4489aa55dc5bc80c58711|0.7497 |1.0E-4  |0.001      |
|3edde31d5d6c4994bb6f55a385ecbe0b|0.7504 |0.001   |0.001      |
|ff9243617b2a423faa53b04bb48b0f89|0.7585 |0.01    |0.001      |
|c9aba721e4324bc68980e66260fd6fe6|0.7758 |0.1     |0.001      |
|1d88d9c3f7754cfc9c80b4fe14ffb059|0.7874 |0.5     |1.0E-4     |
|6a8ed4640640451d82265ce45e3b45b7|0.7875 |0.5     |0.001      |
|0041a8f619434408b2fb8d424292eddc|0.7883 |0.5     |0.01       |
|c5ab9b9e8401491b9acf5a092de29dec|0.7948 |0.5     |0.1        |
|d82c0e7b8c214db3affdc48519a4defd|0.799  |1.0     |0.001      |
|9f101be55ec44116b03ebbcd1abecf47|0.8222 |0.5     |0.5        |
+--------------------------------+-------+--------+-----------+
```


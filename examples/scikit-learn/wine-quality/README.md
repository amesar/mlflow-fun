# mlflow-fun - scikit-learn - Wine Quality Example

## Overview
* Wine Quality Elastic Net Example
* This example demonstrates all features of MLflow training and prediction.
* Saves model in pickle format
* Saves text and plot artifacts
* Shows several ways to run the training - use _mlflow run_, run against Databricks cluster, etc.
* Shows several ways to run the prediction  - web server,  mlflow.load_model(), UDF, etc.

## Setup

```
pip install mlflow
pip install cloudpickle
pip install pyarrow # for Spark UDF example
pip install databricks-cli  # for Databricks examples
```

## Training

Source: [main_train_wine_quality.py](main_train_wine_quality.py) and [train_wine_quality.py](wine_quality/train_wine_quality.py).

### Standard Python Main Run

To run with standard main function:
```
python main_train_wine_quality.py  0.5 0.5 wine-quality.csv
```

### Project Runs

These runs use the [MLproject](MLproject) file. For more details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

**mlflow local**
```
mlflow run . -Palpha=0.01 -Pl1_ratio=0.75 -Prun_origin=LocalRun
```

**mlflow github**
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/scikit-learn/wine-quality \
  -Palpha=0.01 -Pl1_ratio=0.75 -Prun_origin=GitRun
```

**mlflow Databricks remote** - Run against Databricks. See [Remote Execution on Databricks](https://mlflow.org/docs/latest/projects.html#remote-execution-on-databricks) and [mlflow_run_cluster.json](mlflow_run_cluster.json).

Setup:
```
export MLFLOW_TRACKING_URI=databricks
export DATABRICKS_TOKEN=MY_TOKEN
export DATABRICKS_HOST=https://acme.cloud.databricks.com
```
Now run:
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/scikit-learn/wine-quality \
  -Palpha=0.01 -Pl1_ratio=0.75 -Prun_origin=GitRun \
  -Pdata_path=/dbfs/tmp/data/wine-quality.csv \
  -m databricks --cluster-spec mlflow_run_cluster.json
```

### Databricks Cluster Runs

You can also package your code as an egg and run it with the standard Databricks REST API run endpoints.
See [runs submit](https://docs.databricks.com/api/latest/jobs.html#runs-submit), [run now](https://docs.databricks.com/api/latest/jobs.html#run-now) and [spark_python_task](https://docs.databricks.com/api/latest/jobs.html#jobssparkpythontask). In this example we use runs_submit.

Setup.
```
pip install databricks-cli
```

Build the egg.
```
python setup.py bdist_egg
```

Upload the data file, main file and egg to your Databricks cluster.
```
databricks fs cp main_train_wine_quality.py dbfs:/tmp/jobs/wine_quality/main.py
databricks fs cp wine-quality.csv dbfs:/tmp/jobs/wine_quality/wine-quality.csv
databricks fs cp \
  dist/mlflow_wine_quality-0.0.1-py3.6.egg \
  dbfs:/tmp/jobs/wine_quality/mlflow_wine_quality-0.0.1-py3.6.egg
```

#### Run with new cluster
Create [run_submit_new_cluster.json](run_submit_new_cluster.json). Some extracts:
```
  "libraries": [
    { "pypi": { "package": "mlflow" } },
    { "pypi": { "package": "cloudpickle" }},
    { "egg": "dbfs:/tmp/jobs/wine_quality/mlflow_wine_quality-0.0.1-py3.6.egg" }
  ],
  "spark_python_task": {
    "python_file": "dbfs:/tmp/jobs/wine_quality/main.py",
    "parameters": [ 0.5, 0.5, "/dbfs/tmp/jobs/wine_quality/wine-quality.csv", "run_submit_egg" ]
  },
```

Launch run.
```
curl -X POST -H "Authorization: Bearer MY_TOKEN" \
  -d @run_submit_new_cluster.json  \
  https://acme.cloud.databricks.com/api/2.0/jobs/runs/submit
```

#### Run with existing cluster


Attach the egg to the cluster and restart cluster.
```
databricks libraries install --cluster-id 1222-015510-grams64 --egg dbfs:/tmp/jobs/wine_quality/mlflow_wine_quality-0.0.1-py3.6.egg
databricks clusters restart --cluster-id 1222-015510-grams64
```

Create [run_submit_existing_cluster.json](run_submit_existing_cluster.json). 
```
  "run_name": "MLflow_RunSubmit_ExistingCluster",
  "existing_cluster_id": "1222-015510-grams64",
  "timeout_seconds": 3600,
  "spark_python_task": {
    "python_file": "dbfs:/tmp/jobs/wine_quality/main.py",
    "parameters": [ 0.3, 0.3, "/dbfs/tmp/jobs/wine_quality/wine-quality.csv", "run_submit_egg" ]
  }
```
Launch run.
```
curl -X POST -H "Authorization: Bearer MY_TOKEN" \
  -d @run_submit_existing_cluster.json  \
  https://acme.cloud.databricks.com/api/2.0/jobs/runs/submit
```

## Predictions

You can make predictions in the following ways:
1. Use MLflow's serving web server and submit predictions via HTTP
2. Call mlflow.sklearn.load_model() from your own serving code and then make predictions
4. Call mlflow.pyfunc.load_pyfunc() from your own serving code and then make predictions
5. Batch prediction with Spark UDF (user-defined function)


See MLflow documentation:
* [Tutorial - Serving the Model](https://www.mlflow.org/docs/latest/tutorial.html#serving-the-model)
* [Quickstart - Saving and Serving Models](https://www.mlflow.org/docs/latest/quickstart.html#saving-and-serving-models)
* [mlflow.pyfunc.spark_udf](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf)


### Data for predictions
[wine-quality.json](wine-quality.json):
```
[
  {
    "fixed acidity": 7,
    "volatile acidity": 0.27,
    "citric acid": 0.36,
    "residual sugar": 20.7,
    "chlorides": 0.045,
    "free sulfur dioxide": 45,
    "total sulfur dioxide": 170,
    "density": 1.001,
    "pH": 3,
    "sulphates": 0.45,
    "alcohol": 8.8
  }, 
  . . . . .
]
```

### 1. Serving Models from MLflow Web Server

In one window run the server.
```
mlflow pyfunc serve -p 5001 -r 7e674524514846799310c41f10d6b99d -m model
```

In another window, submit a prediction.
```
curl -X POST -H "Content-Type:application/json" -d @wine-quality.json http://localhost:5001/invocations

[
    5.551096337521979,
    5.297727513113797,
    5.427572126267637,
    5.562886443251915,
    5.562886443251915
]
```

### 2. Predict with mlflow.sklearn.load_model()

```
python scikit_predict.py 7e674524514846799310c41f10d6b99d

predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [scikit_predict.py](scikit_predict.py):
```
model = mlflow.sklearn.load_model("model",run_id="7e674524514846799310c41f10d6b99d")
df = pd.read_json("wine-quality.json")
predicted = model.predict(df)
print("predicted:",predicted)
```

### 3. Predict with mlflow.pyfunc.load_pyfunc()

```
python pyfunc_predict.py 7e674524514846799310c41f10d6b99d

predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [pyfunc_predict.py](pyfunc_predict.py):
```
model_uri = mlflow.start_run("7e674524514846799310c41f10d6b99d").info.artifact_uri +  "/model"
model = mlflow.pyfunc.load_pyfunc(model_uri)
df = pd.read_json("wine-quality.json")
predicted = model.predict(df)
print("predicted:",predicted)
```

### 4. Batch prediction with Spark UDF (user-defined function)

Scroll right to see prediction column.

```
pip install pyarrow

spark-submit --master local[2] spark_udf_predict.py 7e674524514846799310c41f10d6b99d

+-------+---------+-----------+-------+-------------+-------------------+----+--------------+---------+--------------------+----------------+------------------+
|alcohol|chlorides|citric acid|density|fixed acidity|free sulfur dioxide|  pH|residual sugar|sulphates|total sulfur dioxide|volatile acidity|        prediction|
+-------+---------+-----------+-------+-------------+-------------------+----+--------------+---------+--------------------+----------------+------------------+
|    8.8|    0.045|       0.36|  1.001|          7.0|               45.0| 3.0|          20.7|     0.45|               170.0|            0.27| 5.551096337521979|
|    9.5|    0.049|       0.34|  0.994|          6.3|               14.0| 3.3|           1.6|     0.49|               132.0|             0.3| 5.297727513113797|
|   10.1|     0.05|        0.4| 0.9951|          8.1|               30.0|3.26|           6.9|     0.44|                97.0|            0.28| 5.427572126267637|
|    9.9|    0.058|       0.32| 0.9956|          7.2|               47.0|3.19|           8.5|      0.4|               186.0|            0.23| 5.562886443251915|
```
From [spark_udf_predict.py](spark_udf_predict.py):
```
spark = SparkSession.builder.appName("ServePredictions").getOrCreate()
df = spark.read.option("inferSchema",True).option("header", True).csv("wine-quality.csv")
df = df.drop("quality")

udf = mlflow.pyfunc.spark_udf(spark, "model", run_id="7e674524514846799310c41f10d6b99d")
df2 = df.withColumn("prediction", udf(*df.columns))
df2.show(10)
```

### 5. Unpickle model artifact file without MLflow and predict
You can directly read the model pickle file and then make predictions.
From [pickle_predict.py](pickle_predict.py):
```
pickle_path = "/opt/mlflow/mlruns/3/11df004981b443908d9286d54d24dc27/artifacts/model/model.pkl"
with open(pickle_path, 'rb') as f:
    model = pickle.load(f)
df = pd.read_json("wine-quality.json")
predicted = model.predict(df)
print("predicted:",predicted)
```

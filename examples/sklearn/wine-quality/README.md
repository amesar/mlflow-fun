# mlflow-fun - sklearn - Wine Quality Example

## Overview
* Data -  Wine Quality
* Model -  DecisionTreeRegressor
* Synopsis
  * This example demonstrates many features of MLflow training and prediction
  * Source: [train.py](wine_quality/train.py) and [predict.py](predict.py)
  * Saves model in pickle format
  * Saves plot artifact
  * Package structure so egg can be deployed an run in Databricks environment
* Shows several ways to run training:
  * _mlflow run_ CLI 
  * run against Databricks cluster using REST API
  * call egg from notebook
* Shows several ways to run predictions
  * web server
  * mlflow.sklearn.load_model()
  * mlflow.pyfunc.load_pyfunc()
  * mlflow.pyfunc.spark_udf 
* Data sets: ../../data/wine-quality - wine-quality-white.csv and wine-quality-red.csv.

## Setup

```
pip install mlflow
pip install matplotlib
pip install pyarrow # for Spark UDF example
```

## Training

Source: [main.py](main.py) and [train.py](wine_quality/train.py).

### Unmanaged without mlflow run

#### Command-line python

To run with standard main function:
```
python main.py --experiment_name sklearn_wine \
  --max_depth 2 --max_leaf_nodes 32 \
  --data_path ../../data/wine-quality/wine-quality-white.csv 
```

#### Jupyter notebook
See [Train_Wine_Quality.ipynb](Train_Wine_Quality.ipynb).
```
export MLFLOW_TRACKING_URI=http://localhost:5000
jupyter notebook
```

### Using mlflow run

These runs use the [MLproject](MLproject) file. For more details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Note that mlflow run ignores the `set_experiment()` function so you must specify the experiment with the  `--experiment-id` argument.

**mlflow run local**
```
mlflow run . -P max_depth=2 -P max_leaf_nodes=32 -P run_origin=mlflow_run_local \
  -P data_path=../../data/wine-quality/wine-quality-white.csv --experiment-id=2019
```

**mlflow run github**
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/scikit-learn/wine-quality \
  -P max_depth=2 -P max_leaf_nodes=32 -P run_origin=mlflow_run_git \
  -P data_path=https://raw.githubusercontent.com/amesar/mlflow-fun/master/examples/data/wine-quality/wine-quality-white.csv \
  --experiment-id=2019
```

**mlflow run Databricks remote** - Run against Databricks. 

See [Remote Execution on Databricks](https://mlflow.org/docs/latest/projects.html#remote-execution-on-databricks) and [mlflow_run_cluster.json](mlflow_run_cluster.json).

Setup.
```
export MLFLOW_TRACKING_URI=databricks
```
The token and tracking server URL will be picked up from your Databricks CLI ~/.databrickscfg default profile.

Now run.
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/scikit-learn/wine-quality \
  -P max_depth=2 -P max_leaf_nodes=32 -P run_origin=mlflow_run_git_dbx \
  -P data_path=/dbfs/tmp/data/wine-quality-white.csv \
  --experiment-id=2019 \
  --mode databricks --cluster-spec mlflow_run_cluster.json
```

### Databricks Cluster Runs

You can also package your code as an egg and run it against a Databricks cluster using the [Databricks CLI](https://docs.databricks.com/user-guide/dev-tools/databricks-cli.html).

#### Setup

Build the egg.
```
python setup.py bdist_egg
```

Upload the data file, main file and egg to DBFS.
```
databricks fs cp main.py dbfs:/tmp/jobs/wine_quality/main.py
databricks fs cp ../../data/wine-quality/wine-quality-white.csv dbfs:/tmp/jobs/wine_quality/wine-quality-white.csv
databricks fs cp \
  dist/mlflow_wine_quality-0.0.1-py3.6.egg \
  dbfs:/tmp/jobs/wine_quality/mlflow_wine_quality-0.0.1-py3.6.egg
```


#### Run Submit

##### Run with new cluster

Define your run in [run_submit_new_cluster.json](run_submit_new_cluster.json) and launch the run.

```
databricks runs submit --json-file run_submit_new_cluster.json
```

##### Run with existing cluster

Every time you build a new egg, you need to upload (as described above) it to DBFS and restart the cluster.
```
databricks clusters restart --cluster-id 1222-015510-grams64
```

Define your run in [run_submit_existing_cluster.json](run_submit_existing_cluster.json) and launch the run.
```
databricks runs submit --json-file run_submit_existing_cluster.json
```

#### Job Run Now

##### Run with new cluster

First create a job with the spec file [create_job_new_cluster.json](create_job_new_cluster.json). 
```
databricks jobs create --json-file create_job_new_cluster.json
```

Then run the job with desired parameters.
```
databricks jobs run-now --job-id $JOB_ID --python-params ' [ "WineQualityExperiment", 0.3, 0.3, "/dbfs/tmp/jobs/wine_quality/wine-quality-white.csv" ] '
```

##### Run with existing cluster
First create a job with the spec file [create_job_existing_cluster.json](create_job_existing_cluster.json).
```
databricks jobs create --json-file create_job_existing_cluster.json
```

Then run the job with desired parameters.
```
databricks jobs run-now --job-id $JOB_ID --python-params ' [ "WineQualityExperiment", 0.3, 0.3, "/dbfs/tmp/jobs/wine_quality/wine-quality-white.csv" ] '
```


#### Run egg from Databricks notebook

Create a notebook with the following cell. Attach it to the existing cluster described above.
```
from wine_quality import Trainer
data_path = "/dbfs/tmp/jobs/wine_quality/wine-quality-white.csv"
trainer = Trainer("WineQualityExperiment", data_path, "from_notebook_with_egg")
trainer.train(2, 32)
```

## Predictions

You can make predictions in the following ways:
1. Use MLflow's serving web server and submit predictions via HTTP calls
2. Call mlflow.sklearn.load_model() from your own serving code and then make predictions
4. Call mlflow.pyfunc.load_pyfunc() from your own serving code and then make predictions
5. Batch prediction with Spark UDF (user-defined function)


See MLflow documentation:
* [Tutorial - Serving the Model](https://www.mlflow.org/docs/latest/tutorial.html#serving-the-model)
* [Quickstart - Saving and Serving Models](https://www.mlflow.org/docs/latest/quickstart.html#saving-and-serving-models)
* [mlflow.pyfunc.spark_udf](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf)


### 1. Serving Models from MLflow Web Server

In one window run the server.
```
mlflow models serve --port 5001 --model-uri runs:/7e674524514846799310c41f10d6b99d/sklearn-model
```

In another window, submit a prediction from [predict-wine-quality.json](../../data/wine-quality/predict-wine-quality.json).
```
curl -X POST -H "Content-Type:application/json" \
  -d @../../data/wine-quality/predict-wine-quality.json \
  http://localhost:5001/invocations
```
```
[
    5.915754923413567
]
```

### 2. Serving Models from Docker Container

First build the docker image.
```
mlflow models build-docker \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/sklearn-model \
  --name sklearn-wine-server
```

Then launch the server as a  docker container.
```
docker run --p 5001:8000 sklearn-wine-server
```

Make predictions with curl as in the step above.

### 3. Serving Models from Sagemaker Docker Container

First build the docker image.
```
mlflow sagemaker build-and-push-container --build --no-push --container sm-sklearn-wine-server
```

Then launch the server as a  docker container.
```
mlflow sagemaker run-local \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/sklearn-model \
  --port 5001 --image sm-sklearn-wine-server
```

Make predictions with curl as in the step above.

### 4. Predict with mlflow.sklearn.load_model()

```
python scikit_predict.py 7e674524514846799310c41f10d6b99d

predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [scikit_predict.py](scikit_predict.py):
```
model = mlflow.sklearn.load_model(f"runs:/7e674524514846799310c41f10d6b99d/sklearn-model")
df = pd.read_csv("../../data/wine-quality/wine-quality-white.csv")
predicted = model.predict(df)
print("predicted:",predicted)
```

### 5. Predict with mlflow.pyfunc.load_pyfunc()

```
python pyfunc_predict.py 7e674524514846799310c41f10d6b99d

predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [pyfunc_predict.py](pyfunc_predict.py):
```
model_uri = mlflow.start_run("7e674524514846799310c41f10d6b99d").info.artifact_uri +  "/sklearn-model"
model = mlflow.pyfunc.load_pyfunc(model_uri)
df = pd.read_csv("../../data/wine-quality/wine-quality-white.csv")
predicted = model.predict(df)
print("predicted:",predicted)
```

### 6. UDF Predict with  mlflow.pyfunc.spark_udf()

From [spark_udf_predict.py](spark_udf_predict.py).

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
df = spark.read.option("inferSchema",True).option("header", True).\
    csv("../../data/wine-quality/wine-quality-white.csv")
df = df.drop("quality")

udf = mlflow.pyfunc.spark_udf(spark, f"runs:/{run_id}/sklearn-model")
df = df.withColumn("prediction", udf(*df.columns))
df.show(10)
```

### 7. Unpickle model artifact file without MLflow and predict
You can directly read the model pickle file and then make predictions.
From [pickle_predict.py](pickle_predict.py):
```
pickle_path = "/opt/mlflow/mlruns/3/11df004981b443908d9286d54d24dc27/artifacts/sklearn-model/model.pkl"
with open(pickle_path, 'rb') as f:
    model = pickle.load(f)
df = pd.read_csv("../../data/wine-quality/wine-quality-white.csv")
predicted = model.predict(df)
print("predicted:",predicted)
```

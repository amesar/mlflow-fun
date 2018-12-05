# mlflow-fun Scikit-learn Examples

## Overview

*  Saves text and plot artifacts
*  Saves models in pickle format
*  Serves models with mlflow.load_model() or MLflow serving web server

## Initialization

Set the URI of your MLflow tracking server.
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Wine Quality Elastic Net Example

This example demonstrates most of the features of MLflow training and prediction.
```
cd wine-quality
```

### Training

Source: [train_wine_quality.py](wine-quality/train_wine_quality.py).

#### Standard Main Run

To run with standard main function:
```
python train_wine_quality.py wine.csv 0.5 0.5
```

Check results in UI:
```
http://localhost:5011/#/experiments/1
```

#### Managed Runs

To run locally with the [MLproject](iris/MLproject) file:
```
mlflow run . -Palpha=0.01 -Pl1_ratio=0.75 -Prun_origin=LocalRun
```

To run from git with the [MLproject](iris/MLproject) file:
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/scikit-learn/wine-quality -Palpha=0.01 -Pl1_ratio=0.75 -Prun_origin=GitRun
```

### Predictions

You can make predictions in the following ways:
1. Use MLflow's serving web server and submit predictions via HTTP
2. Call mlflow.sklearn.load_model() from your own serving code and then make predictions
4. Call mlflow.pyfunc.load_pyfunc() from your own serving code and then make predictions
5. Batch prediction with Spark UDF (user-defined function)


See MLflow documentation:
* [Tutorial - Serving the Model](https://www.mlflow.org/docs/latest/tutorial.html#serving-the-model)
* [Quickstart - Saving and Serving Models](https://www.mlflow.org/docs/latest/quickstart.html#saving-and-serving-models)
* [mlflow.pyfunc.spark_udf](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf)


#### Data for predictions
[wine-quality.json](wine-quality/wine-quality.json):
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

#### 1. Serving Models from MLflow Web Server

In one window run the server:
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

#### 2. Predict with mlflow.sklearn.load_model()

```
python scikit_predict.py 7e674524514846799310c41f10d6b99d

predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [scikit_predict.py](wine-quality/scikit_predict.py):
```
model = mlflow.sklearn.load_model("model",run_id="7e674524514846799310c41f10d6b99d")
with open("wine-quality.json", 'rb') as f:
    data = json.loads(f.read())
predicted = model.predict(df)
print("predicted:",predicted)
```

#### 3. Predict with mlflow.pyfunc.load_pyfunc()

```
python pyfunc_predict.py 7e674524514846799310c41f10d6b99d

predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [pyfunc_predict.py](wine-quality/pyfunc_predict.py):
```
model_uri = mlflow.start_run("7e674524514846799310c41f10d6b99d").info.artifact_uri +  "/model"
model = mlflow.pyfunc.load_pyfunc(model_uri)
with open("wine-quality.json", 'rb') as f:
    data = json.loads(f.read())
predicted = model.predict(df)
print("predicted:",predicted)
```

#### 4. Batch prediction with Spark UDF (user-defined function)

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
From [spark_udf_predict.py](wine-quality/spark_udf_predict.py):
```
path = "wine-quality.csv"
spark = SparkSession.builder.appName("ServePredictions").getOrCreate()
df = spark.read.option("inferSchema",True).option("header", True).csv(path)
df = df.drop("quality")

udf = mlflow.pyfunc.spark_udf(spark, "model", run_id="7e674524514846799310c41f10d6b99d")
df2 = df.withColumn("prediction", udf(*df.columns))
df2.show(10)
```


## Iris Decision Tree Example

Simple Scikit-learn [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/tree.html) that:
* Logs parameters and metrics
* Saves text artifacts: confusion_matrix.txt and classification_report.txt
* Saves plot artifact: simple_plot.png
* Saves model as a pickle file

Source: [train_iris_decision_tree.py](iris/train_iris_decision_tree.py)

To run with standard main function:
```
cd iris
python train_iris_decision_tree.py 5 3
```

To run locally with the [MLproject](iris/MLproject) file:
```
mlflow run . -Pmin_samples_leaf=5 -Pmax_depth=3
```

To run from git with the [MLproject](iris/MLproject) file:
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/scikit-learn/iris -Pmin_samples_leaf=5 -Pmax_depth=3 -Ptag=RunFromGit
```

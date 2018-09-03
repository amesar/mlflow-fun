# mlflow-fun

MLflow examples for Python and Scala training models.

Examples:
* Python scikit-learn classifiers showing:
  *  Saving text and plot artifacts 
  *  Saving models in pickle format
  *  Serving models with mlflow.load_model() or MLflow serving web server
* Spark Scala classifier using MLFlow Java client.

## Install

Install mlflow either with PyPi or Miniconda.

### 1. PyPi

Install MLflow from PyPi: ``pip install mlflow``

### 2. Miniconda

**Install miniconda3**
```
https://conda.io/miniconda.html
```

**Create environment**
```
conda env create --file conda.yaml
```
**Source environment**
```
source activate mlflow-fun
```

## Run Server

**Launch server**
```
mlflow server --host 0.0.0.0 
```
## Run Samples

### Python Scikit-learn samples

Set the URI of your MLflow tracking server:
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```

#### Wine Quality Elastic Net

Source: [train_wine_quality.py](examples/scikit-learn/wine-quality/train_wine_quality.py).

To run with standard main function:
```
cd examples/scikit-learn/wine-quality
python train_wine_quality.py wine.csv 0.5 0.5
```

To run with the [MLproject](examples/scikit-learn/wine-quality/MLproject) file:
```
mlflow run . -Palpha=0.01 -Pl1_ratio=0.75 
```

Check results in UI:
```
http://localhost:5011/#/experiments/1
```

**Serving Models**

You can serve a specific model run in two ways:
* Use MLflow's serving web server and submit predictions via HTTP
* Call load_model() from your own serving code and then make predictions


See MLflow documentation:
* [Tutorial - Serving the Model](https://www.mlflow.org/docs/latest/tutorial.html#serving-the-model)
* [Quickstart - Saving and Serving Models](https://www.mlflow.org/docs/latest/quickstart.html#saving-and-serving-models)

**MLflow Model Serving Web Server**

In one window run the server:
```
mlflow sklearn serve -p 5001 -r 7e674524514846799310c41f10d6b99d
```

In another window, submit a prediction.
```
curl -X POST -H "Content-Type:application/json" -d @predictions.json http://localhost:5001/invocations

```
[predictions.json](examples/scikit-learn/wine-quality/predictions.json):
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
  }
]
```

**Custom Serving with load_model()**

Run:
```
python serve_loaded_model.py 7e674524514846799310c41f10d6b99d
```
Using ``mlflow.sklearn.load_model()`` as in [serve_loaded_model.py](examples/scikit-learn/wine-quality/serve_loaded_model.py):
```
clf = mlflow.sklearn.load_model("model",run_id="7e674524514846799310c41f10d6b99d")
with open("predictions.json", 'rb') as f:
    data = json.loads(f.read())
df = json_normalize(data)
predicted = clf.predict(df)
print("predicted:",predicted)
```

#### Iris Decision Tree

Simple Scikit-learn [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/tree.html) that:
* Logs parameters and metrics 
* Saves text artifacts: confusion_matrix.txt and classification_report.txt
* Saves plot artifact: simple_plot.png
* Saves model as a pickle file

Source: [train_iris_decision_tree.py](examples/scikit-learn/iris/train_iris_decision_tree.py)

To run with standard main function:
```
cd examples/scikit-learn/iris
python train_iris_decision_tree.py 5 3
```

To run with the [MLproject](examples/scikit-learn/iris/MLproject) file:
```
mlflow run . -Pmin_samples_leaf=5 -Pmax_depth=3
```


### Scala Spark ML sample

**Install MLflow Java Client**

Until the Java client is pushed to Maven central, install it locally.

```
git clone -b java-client https://github.com/mlflow/mlflow mlflow-java-client
cd mlflow-java-client/mlflow/java
mvn -DskipTests=true install
```

More details at: [https://github.com/mlflow/mlflow/tree/java-client/mlflow/java/client](https://github.com/mlflow/mlflow/tree/java-client/mlflow/java/client).

**Sample Source Snippet**

Source: [DecisionTreeRegressionExample.scala](examples/spark-scala/src/main/scala/org/andre/mlflow/examples/DecisionTreeRegressionExample.scala)
```
import org.mlflow.client.ApiClient
import org.mlflow.client.objects.ObjectUtils
import org.mlflow.api.proto.Service.{RunStatus,SourceType}

val mlflowClient = ApiClient.fromTrackingUri("http://localhost:5000")

// MLflow - create or get existing experiment
val expName = "scala/SimpleDecisionTreeRegression"
val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)

// MLflow - create run
val sourceName = getClass().getSimpleName()+".scala"
val request = ObjectUtils.makeCreateRun(expId, "MyScalaRun", SourceType.LOCAL, sourceName, System.currentTimeMillis(), "doe")
val runInfo = mlflowClient.createRun(request)
val runId = runInfo.getRunUuid()

. . .

// MLflow - Log parameters
mlflowClient.logParameter(runId, "maxDepth",""+dt.getMaxDepth)
mlflowClient.logParameter(runId, "maxBins",""+dt.getMaxBins)

. . .

// MLflow - Log metric
mlflowClient.logMetric(runId, "rmse",rmse.toFloat)

// MLflow - close run
mlflowClient.updateRun(runId, RunStatus.FINISHED, System.currentTimeMillis())
```
**Run sample**

```
cd examples/spark-scala
mvn package
spark-submit \
  --class org.andre.mlflow.examples.DecisionTreeRegressionExample \
  --master local[2] \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000 \
  data/sample_libsvm_data.txt

Experiment name: scala/DecisionTreeRegressionExample
Experiment ID: 2

```
**Check Results in UI**
```
http://localhost:5000/experiments/2
```
### Scala Spark ML sample using older deprecated Jackson-based MLflow client

**Install MLflow Java Client**

Until the Java client is pushed to Maven central, install it locally.

```
git clone -b java-client-jackson https://github.com/amesar/mlflow mlflow-java-client-jackson
cd mlflow-java-client-jackson
mvn -DskipTests=true install
```

More details at: [https://github.com/amesar/mlflow/tree/java-client-jackson/mlflow-java](https://github.com/amesar/mlflow/tree/java-client-jackson/mlflow-java).

**Run sample**

Source: [DecisionTreeRegressionExample.scala](examples/spark-scala-jackson/src/main/scala/org/andre/mlflow/examples/DecisionTreeRegressionExample.scala)
```
cd examples/spark-scala-jackson
mvn package
spark-submit \
  --class org.andre.mlflow.examples.DecisionTreeRegressionExample \
  --master local[2] \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000 \
  data/sample_libsvm_data.txt

Experiment name: scala/DecisionTreeRegressionExample
Experiment ID: 2

```
**Check Results in UI**
```
http://localhost:5000/experiments/2
```

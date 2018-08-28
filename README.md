# mlflow-fun

MLflow examples for Python and Scala training models.

## Install

Install mlflow in either of two ways.

### 1. PyPi

Install MLflow from PyPi: ``pip install mlflow``

### 2. or miniconda

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

### Python sklearn sample

**Run sample**

Simple Scikit-learn [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/tree.html) that:
* Logs parameters and metrics 
* Saves text artifacts: confusion_matrix.txt and classification_report.txt
* Saves plot artifact: simple_plot.png
* Saves model as a pickle file

Source: [iris_decision_tree.py](examples/sklearn/iris_decision_tree.py).

```
export MLFLOW_TRACKING_URI=http://localhost:5000
cd examples/sklearn
python iris_decision_tree.py
```
**Check Results in UI**
```
http://localhost:5000/experiments/1
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

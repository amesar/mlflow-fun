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

Source: [iris_decision_tree.py](examples/sklearn/iris_decision_tree.py)

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
git clone -b java-client-jackson https://github.com/amesar/mlflow mlflow-java-client-jackson
cd mlflow-java-client-jackson
mvn -DskipTests=true install
```

More details at: [https://github.com/amesar/mlflow/tree/java-client-jackson/mlflow-java](https://github.com/amesar/mlflow/tree/java-client-jackson/mlflow-java).

**Run sample**

Source: [DecisionTreeRegressionExample.scala](examples/spark-scala/src/main/scala/org/andre/mlflow/examples/DecisionTreeRegressionExample.scala)
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

# mlflow-fun - Spark Scala Example

## Install MLflow Java Client

Until the Java client jar is pushed to Maven central, you will need to install the jar locally.

```
git clone -b java-client https://github.com/mlflow/mlflow mlflow
cd mlflow/mlflow/java
mvn -DskipTests=true install
```

More details at: [https://github.com/mlflow/mlflow/tree/java-client/mlflow/java/client](https://github.com/mlflow/mlflow/tree/java-client/mlflow/java/client).

## Run sample

```
cd examples/spark-scala
mvn package
spark-submit \
  --class org.andre.mlflow.examples.DecisionTreeRegressionExample \
  --master local[2] \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000 \
  data/sample_libsvm_data.txt \
  5 5

Experiment name: scala/DecisionTreeRegressionExample
Experiment ID: 2

```
Then check results in UI:
```
http://localhost:5000/experiments/2
```

## Source

Source snippet from [DecisionTreeRegressionExample.scala](src/main/scala/org/andre/mlflow/examples/DecisionTreeRegressionExample.scala).
```
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus

val mlflowClient = new MlflowClient("http://localhost:5000")

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
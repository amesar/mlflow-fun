# mlflow-fun - Spark Scala Example

## Build
```
mvn clean package
```

## Samples

### Quick Start Sample
#### Run
```
spark-submit \
  --class org.andre.mlflow.examples.QuickStart \
  --master local[2] \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000 \
```

#### Source
Source snippet from [QuickStart.scala](src/main/scala/org/andre/mlflow/examples/QuickStart.scala).
```
// Create client
val trackingUri = args(0)
val mlflowClient = new MlflowClient(trackingUri)

// Create or get existing experiment
val expName = "scala/QuickStart"
val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)
println("Experiment name: "+expName)
println("Experiment ID: "+expId)

// Create run
val sourceName = getClass().getSimpleName()+".scala"
val runInfo = mlflowClient.createRun(expId, sourceName);
val runId = runInfo.getRunUuid()

// Log params and metrics
mlflowClient.logParam(runId, "p1","hi")
mlflowClient.logMetric(runId, "m1",0.123F)

// Close run
mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
```

### Spark ML DecisionTreeRegressionExample Sample
#### Run
```
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

#### Source

Source snippet from [DecisionTreeRegressionExample.scala](src/main/scala/org/andre/mlflow/examples/DecisionTreeRegressionExample.scala).
```
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus

// Create client
val mlflowClient = new MlflowClient("http://localhost:5000")

// MLflow - create or get existing experiment
val expName = "scala/SimpleDecisionTreeRegression"
val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)

// MLflow - create run
val sourceName = getClass().getSimpleName()+".scala"
val runInfo = mlflowClient.createRun(expId, sourceName);
val runId = runInfo.getRunUuid()

. . .

// MLflow - Log parameters
mlflowClient.logParameter(runId, "maxDepth",""+dt.getMaxDepth)
mlflowClient.logParameter(runId, "maxBins",""+dt.getMaxBins)

. . .

// MLflow - Log metric
mlflowClient.logMetric(runId, "rmse",rmse.toFloat)

// MLflow - close run
mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
```

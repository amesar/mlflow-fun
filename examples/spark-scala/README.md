# mlflow-fun - Spark Scala Example

## Build
```
mvn clean package
```

## Samples

### Hello World 
#### Run
```
spark-submit \
  --class org.andre.mlflow.examples.HelloWorld \
  --master local[2] \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000 \
```

#### Source
Source snippet from [HelloWorld.scala](src/main/scala/org/andre/mlflow/examples/HelloWorld.scala).
```
// Create client
val trackingUri = args(0)
val mlflowClient = new MlflowClient(trackingUri)

// Create or get existing experiment
val expName = "scala/HelloWorld"
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
Saves model as artifact in MLflow (using temporary directory 'tmp').
#### Run
```
rm -rf tmp
spark-submit \
  --class org.andre.mlflow.examples.DecisionTreeRegressionExample \
  --master local[2] \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --dataPath data/sample_libsvm_data.txt \
  --maxDepth 5 --maxBins 5

Experiment name: scala/DecisionTreeRegressionExample
Experiment ID: 2
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

// MLflow - save model as artifact
//pipeline.save("tmp")
clf.save("tmp")
mlflowClient.logArtifacts(runId, new File("tmp"),"model")

// MLflow - close run
mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
```

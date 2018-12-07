# mlflow-fun - Spark Scala Example

## Build
```
mvn clean package
```

## Hello World Sample
### Run
```
spark-submit \
  --class org.andre.mlflow.examples.HelloWorld \
  --master local[2] \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000 \
```

### Source
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

## Spark ML DecisionTree Sample

Sample demonstrating:
*  Training a model
*  Saving the model
*  Predicting from loading model

### Train

Saves model as artifact in MLflow.

#### Run
```
spark-submit \
  --class org.andre.mlflow.examples.TrainDecisionTree \
  --master local[2] \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --dataPath data/sample_libsvm_data.txt \
  --modelPath model_sample \
  --maxDepth 5 --maxBins 5

Experiment name: scala/SimpleDecisionTree
Experiment ID: 2
```

#### Source

Source snippet from [TrainDecisionTree.scala](src/main/scala/org/andre/mlflow/examples/TrainDecisionTree.scala).
```
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus

// Create client
val mlflowClient = new MlflowClient("http://localhost:5000")

// MLflow - create or get existing experiment
val expName = "scala/SimpleDecisionTree"
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

### Predict

#### Run
```
spark-submit \
  --class org.andre.mlflow.examples.PredictDecisionTree \
  --master local[2] \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --dataPath data/sample_libsvm_data.txt \
  --modelPath model_sample
```

#### Source

Source snippet from [PredictDecisionTree.scala](src/main/scala/org/andre/mlflow/examples/PredictDecisionTree.scala).
```
val data = spark.read.format("libsvm").load(opts.dataPath)
val model = PipelineModel.load(opts.modelPath)
val predictions = model.transform(data)
println("Prediction:")
predictions.select("prediction", "label", "features").show(10,false)
```

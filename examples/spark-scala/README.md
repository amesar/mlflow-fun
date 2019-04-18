# mlflow-fun - Spark Scala Example

Scala examples using the MLflow Java client:
* Hello World - Simple MLflow example.
* Spark ML DecisionTree - advanced - saves and predicts SparkML and MLeap model formats.

## Build
```
mvn clean package
```

## Hello World Sample
### Run
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.HelloWorld \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000
```
```
Experiment name: scala_HelloWorld
Experiment ID: 3
Run ID: 81cc7941adae4860899ad5449df52802
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
*  Trains a model
*  Saves the model in Spark ML and MLeap formats
*  Predicts from Spark ML and MLeap formats

### Train

Saves model as Spark ML and MLeap artifact in MLflow.


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

// MLflow - save model as Spark ML artifact
val sparkModelPath = "out/spark_model"
model.write.overwrite().save(sparkModelPath)
mlflowClient.logArtifacts(runId, new File(sparkModelPath), "spark_model")

// MLflow - save model as MLeap artifact
val mleapModelDir = new File("out/mleap_model")
mleapModelDir.mkdir
MLeapUtils.save(model, predictions, "file:"+mleapModelDir.getAbsolutePath)
mlflowClient.logArtifacts(runId, mleapModelDir, "mleap_model")

// MLflow - close run
mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
```

### Run against local Spark and local MLflow tracking server

```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.TrainDecisionTree \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --experimentName scala_DecisionTree \
  --dataPath ../data/sample_libsvm_data.txt \
  --modelPath model_sample --maxDepth 5 --maxBins 5
```

### Run against local Spark and Databricks hosted tracking server

```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.TrainDecisionTree \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri https://acme.cloud.databricks.com --token MY_TOKEN \
  --experimentName spark_DecisionTree \
  --dataPath ../data/sample_libsvm_data.txt \
  --modelPath model_sample --maxDepth 5 --maxBins 5
```

### Run in Databricks Cluster

You can also run your jar in a Databricks cluster with the standard Databricks REST API run endpoints.
See [runs submit](https://docs.databricks.com/api/latest/jobs.html#runs-submit), [run now](https://docs.databricks.com/api/latest/jobs.html#run-now) and [spark_jar_task](https://docs.databricks.com/api/latest/jobs.html#jobssparkjartask).
In this example we showcase runs_submit.

#### Setup

Upload the data file and jar to your Databricks cluster.
```
databricks fs cp data/sample_libsvm_data.txt \
  dbfs:/tmp/jobs/spark-scala-example/sample_libsvm_data.txt

databricks fs cp target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  dbfs:/tmp/jobs/spark-scala-example/mlflow-spark-examples-1.0-SNAPSHOT.jar
```

Here is a snippet from
[run_submit_new_cluster.json](run_submit_new_cluster.json) or
[run_submit_existing_cluster.json](run_submit_existing_cluster.json).
```
  "libraries": [
    { "pypi": { "package": "mlflow" } },
    { "jar": "dbfs:/tmp/jobs/spark-scala-example/mlflow-spark-examples-1.0-SNAPSHOT.jar" }
  ],
  "spark_jar_task": {
    "main_class_name": "org.andre.mlflow.examples.TrainDecisionTree",
    "parameters": [ 
      "--dataPath",  "dbfs:/tmp/jobs/spark-scala-example/sample_libsvm_data.txt",
      "--modelPath", "/dbfs/tmp/jobs/spark-scala-example/models",
      "--runOrigin", "run_submit_new_cluster.json"
    ]
  }
```

#### Run with new cluster

Create [run_submit_new_cluster.json](run_submit_new_cluster.json) and launch the run.
```
curl -X POST -H "Authorization: Bearer MY_TOKEN" \
  -d @run_submit_new_cluster.json  \
  https://acme.cloud.databricks.com/api/2.0/jobs/runs/submit
```

#### Run with existing cluster

Every time you build a new jar, you need to upload (as described above) it to DBFS and restart the cluster.
```
databricks clusters restart --cluster-id 0113-005848-about166
```

Create [run_submit_existing_cluster.json](run_submit_existing_cluster.json) and launch the run.
```
curl -X POST -H "Authorization: Bearer MY_TOKEN" \
  -d @run_submit_existing_cluster.json  \
  https://acme.cloud.databricks.com/api/2.0/jobs/runs/submit
```

#### Run jar from Databricks notebook

Create a notebook with the following cell. Attach it to the existing cluster described above.
```
import org.andre.mlflow.examples.TrainDecisionTree
val dataPath = "dbfs:/tmp/jobs/spark-scala-example/sample_libsvm_data.txt"
val modelPath = "/dbfs/tmp/jobs/spark-scala-example/models"
val runOrigin = "run_from_jar_Notebook"
TrainDecisionTree.train(spark, dataPath, modelPath, 5, 5, runOrigin)
```

### Predict

Predicts from Spark ML and MLeap models.

#### Run
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.PredictDecisionTree \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --dataPath data/sample_libsvm_data.txt \
  --runId 3e422c4736a34046a74795384741ac33
```

```
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       0.0|  0.0|(692,[127,128,129...|
|       1.0|  1.0|(692,[158,159,160...|
|       1.0|  1.0|(692,[124,125,126...|
|       1.0|  1.0|(692,[152,153,154...|
+----------+-----+--------------------+
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

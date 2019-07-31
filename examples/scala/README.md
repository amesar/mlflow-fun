# mlflow-fun - Spark Scala Example

Scala training and prediction examples using the MLflow Java client
* [Hello World](#Hello-World) - Simple MLflow example.
* [Resilient Hello World](#Resilient-Hello-World) - Shows how to set run status if run fails.

* Spark ML DecisionTreeRegressor - Advanced - saves and predicts with SparkML and MLeap model formats.

[MLflow tools](#Tools)
* DumpExperiment - Dump experiment as text.
* DumpRun - Dump run as text.
* SearchRuns - Search for runs using search criteria.
* RunsToCsvConverter - Dump experiment runs to CSV file.

## Setup

Note: You must install Python MLflow for MLflow Java client to work: `pip install mlflow`.

## Build
```
mvn clean package
```

## Examples

### Hello World
#### Run
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.hello.HelloWorld \
  target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000
```
```
Experiment name: scala_HelloWorld
Experiment ID: 3
Run ID: 81cc7941adae4860899ad5449df52802
```

### Source
Source snippet from [HelloWorld.scala](src/main/scala/org/andre/mlflow/examples/hello/HelloWorld.scala).
```
// Create client
val trackingUri = args(0)
val mlflowClient = new MlflowClient(trackingUri)

// Create or get existing experiment
val expName = "scala_HelloWorld"
val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)
println("Experiment name: "+expName)
println("Experiment ID: "+expId)

// Create run
val runInfo = mlflowClient.createRun(expId)
val runId = runInfo.getRunId()

// Log params and metrics
mlflowClient.logParam(runId, "p1","hi")
mlflowClient.logMetric(runId, "m1",0.123F)

// Close run
mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
```

### Hello World Context

Uses new MLflow 1.1.0 [MlflowContext](https://mlflow.org/docs/latest/java_api/org/mlflow/tracking/MlflowContext.html).

#### Run
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.hello.HelloWorldContext \
  target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000
```
```
Experiment name: scala_HelloWorld_Context
Experiment ID: 3
Run ID: 81cc7941adae4860899ad5449df52802
```

### Source
Source snippet from [HelloWorld.scala](src/main/scala/org/andre/mlflow/examples/hello/HelloWorld.scala).
```
// Create context and client
val mlflow = new MlflowContext()
val client = mlflow.getClient()

// Create and set experiment
val expName = "scala_HelloWorld_Context"
val expId = MLflowUtils.getOrCreateExperimentId(client, expName)
println("Experiment name: "+expName)
println("Experiment ID: "+expId)
mlflow.setExperimentName(expName)

// Create run
val run = mlflow.startRun("HelloWorld Run")
println("Run ID: "+run.getId)

// Log params and metrics
run.logParam("alpha","0.5")
run.logMetric("rmse",0.876)
run.setTag("origin","laptop")

// End run
run.endRun();
```

### Resilient Hello World

Shows how to set run status if a failure occurs during run. 

#### Run

```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.hello.ResilientHelloWorld \
  target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000 throwException
```
```
Experiment name: scala_ResilientHelloWorld
Experiment ID: 13
Run ID: e3ec538d27e14dbfb05bd2eae01ddb5d
Status FAILED: java.lang.Exception: Ouch
Retrieved run: runId: e3ec538d27e14dbfb05bd2eae01ddb5d - status: FAILED
```
#### Code

Snippet from [ResilientHelloWorld.scala](src/main/scala/org/andre/mlflow/examples/hello/ResilientHelloWorld.scala).
```
  val runId = client.createRun(experimentId).getRunId()
  println("Run ID: "+runId)
  try {
    client.logParam(runId, "alpha", "0.5")
    if (doThrowException) {
      throw new Exception("Ouch")
    }
    client.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
  } catch {
    case e: Exception => {
      client.setTerminated(runId, RunStatus.FAILED, System.currentTimeMillis())
    }
  }
```

### Spark ML DecisionTreeRegressor Sample

Sample that:
*  Trains a model
*  Saves the model in Spark ML and MLeap formats
*  Predicts from Spark ML and MLeap formats
*  Uses wine quality dataset - [wine-quality-white.csv](../data/wine-quality/wine-quality-white.csv)

#### Train

Saves model as Spark ML and MLeap artifact in MLflow.


##### Source

Source snippet from [TrainDecisionTreeRegressor.scala](src/main/scala/org/andre/mlflow/examples/wine/TrainDecisionTreeRegressor.scala).
```
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus

// Create client
val mlflowClient = new MlflowClient("http://localhost:5000")

// MLflow - create or get existing experiment
val expName = "scala_TrainDecisionTreeRegressor"
val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)

// MLflow - create run
val runInfo = mlflowClient.createRun(expId)
val runId = runInfo.getRunUuid()

// MLflow - Log parameters
mlflowClient.logParam(runId, "maxDepth", "1")
mlflowClient.logParam(runId, "maxBins", "32")

. . . 

// MLflow - Log metric
mlflowClient.logMetric(runId, "rmse", 0.786)

// MLflow - save model as artifact
//pipeline.save("tmp")
clf.save("tmp")
mlflowClient.logArtifacts(runId, new File("tmp"),"model")

// MLflow - save model as Spark ML artifact
val sparkModelPath = "out/spark-model"
model.write.overwrite().save(sparkModelPath)
mlflowClient.logArtifacts(runId, new File(sparkModelPath), "spark-model")

// MLflow - close run
mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
```

#### Run against local Spark and local MLflow tracking server

```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.wine.TrainDecisionTreeRegressor \
  target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --experimentName scala_wine \
  --dataPath ../data/wine-quality/wine-quality-white.csv \
  --modelPath model_sample --maxDepth 5 --maxBins 5
```

#### Run against local Spark and Databricks hosted tracking server

```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.wine.TrainDecisionTreeRegressor \
  target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  --trackingUri https://acme.cloud.databricks.com --token MY_TOKEN \
  --experimentName /Shared/experiments/demo/scala_wine \
  --dataPath ../data/wine-quality/wine-quality-white.csv \
  --modelPath model_sample --maxDepth 5 --maxBins 5
```

#### Run in Databricks Cluster

You can also run your jar in a Databricks cluster with the standard Databricks REST API run endpoints.
See [runs submit](https://docs.databricks.com/api/latest/jobs.html#runs-submit), [run now](https://docs.databricks.com/api/latest/jobs.html#run-now) and [spark_jar_task](https://docs.databricks.com/api/latest/jobs.html#jobssparkjartask).
In this example we showcase runs_submit.

##### Setup

Upload the data file and jar to your Databricks cluster.
```
databricks fs cp ../data/wine-quality/wine-quality-white.csv \
  dbfs:/tmp/jobs/scala-example/wine-quality-white.csv

databricks fs cp target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  dbfs:/tmp/jobs/scala-example/mlflow-scala-examples-1.0-SNAPSHOT.jar
```

Here is a snippet from
[run_submit_new_cluster.json](run_submit_new_cluster.json) or
[run_submit_existing_cluster.json](run_submit_existing_cluster.json).
```
  "libraries": [
    { "pypi": { "package": "mlflow" } },
    { "jar": "dbfs:/tmp/jobs/scala-example/mlflow-scala-examples-1.0-SNAPSHOT.jar" }
  ],
  "spark_jar_task": {
    "main_class_name": "org.andre.mlflow.examples.TrainDecisionTree",
    "parameters": [ 
      "--dataPath",  "dbfs:/tmp/jobs/scala-example/wine-quality-white.csv",
      "--modelPath", "/dbfs/tmp/jobs/scala-example/models",
      "--runOrigin", "run_submit_new_cluster.json"
    ]
  }
```

##### Run with new cluster

Create [run_submit_new_cluster.json](run_submit_new_cluster.json) and launch the run.
```
curl -X POST -H "Authorization: Bearer MY_TOKEN" \
  -d @run_submit_new_cluster.json  \
  https://acme.cloud.databricks.com/api/2.0/jobs/runs/submit
```

##### Run with existing cluster

Every time you build a new jar, you need to upload it to DBFS (as described above) and restart the cluster.
```
databricks clusters restart --cluster-id 0113-005848-about166
```

Create [run_submit_existing_cluster.json](run_submit_existing_cluster.json) and launch the run.
```
curl -X POST -H "Authorization: Bearer MY_TOKEN" \
  -d @run_submit_existing_cluster.json  \
  https://acme.cloud.databricks.com/api/2.0/jobs/runs/submit
```

##### Run jar from Databricks notebook

Create a notebook with the following cell. Attach it to the existing cluster described above.
```
import org.andre.mlflow.examples.TrainDecisionTree
val dataPath = "dbfs:/tmp/jobs/scala-example/wine-quality-white.csv"
val modelPath = "/dbfs/tmp/jobs/scala-example/models"
val runOrigin = "run_from_jar_Notebook"
TrainDecisionTree.train(spark, dataPath, modelPath, 5, 5, runOrigin)
```

#### Predict

Predicts from Spark ML and MLeap models. 

There are several ways to obtain the run:
* [PredictByLastRun.scala](src/main/scala/org/andre/mlflow/examples/wine/PredictByLastRun.scala) - Use the latest run.
* [PredictByBestRun.scala](src/main/scala/org/andre/mlflow/examples/wine/PredictByBestRun.scala) - Use the best run for given metric.
* [PredictByRunId.scala](src/main/scala/org/andre/mlflow/examples/wine/PredictByRunId.scala) - Specify run ID.

##### Run
##### Run By RunID
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.wine.PredictByRunId \
  target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --dataPath ../data/wine-quality/wine-quality-white.csv \
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

##### Run By LastRun
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.wine.PredictByLastRun \
  target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --dataPath ../data/wine-quality/wine-quality-white.csv \
  --experimentId 2
```

##### Run By BestRun
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.wine.PredictByBestRun \
  target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --dataPath ../data/wine-quality/wine-quality-white.csv \
  --experimentId 2
  --metric rmse --ascending
```

##### Source

Source snippet from [PredictUtils.scala](src/main/scala/org/andre/mlflow/examples/wine/PredictUtils.scala).
```
val data = spark.read.format("libsvm").load(opts.dataPath) // TODO
val model = PipelineModel.load(opts.modelPath)
val predictions = model.transform(data)
println("Prediction:")
predictions.select("prediction", "label", "features").show(10,false)
```

## Tools

### Text dump tools

Dumps all experiment or run information recursively.

**Overview**
* [DumpRun.scala](src/main/scala/org/andre/mlflow/tools/DumpRun.scala) - Dumps run information.
  * Shows info, params, metrics and tags.
  * Recursively shows all artifacts up to the specified level.
* [DumpExperiment.scala](src/main/scala/org/andre/mlflow/tools/DumpExperiment.scala) - Dumps run information.
  * If `showInfo` is true, then just the run infos will be dumped.
  * If `showData` is true, then an API call for each run will be executed. Beware of experiments with many runs.
* A large value for `artifactMaxLevel` also incurs many API calls.


#### Dump Run
```
scala -cp target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  org.andre.mlflow.tools.DumpRun \
  --runId 033be9f1f7e7494daba64bde62c2cf83 \
  --artifactMaxLevel 2
```
```
RunInfo:
  runId: 033be9f1f7e7494daba64bde62c2cf83
  experimentId: 2
  lifecycleStage: active
  userId: andre
  status: FINISHED
  artifactUri: /opt/mlflow/server/mlruns/2/033be9f1f7e7494daba64bde62c2cf83/artifacts
  startTime: 1561568635358
  endTime:   1561568648413
  startTime: 2019-06-26 13:03:55
  endTime:   2019-06-26 13:04:08
  _duration: 13.055 seconds
Params:
  runOrigin: train.sh_local_env
  maxBins: 32
  maxDepth: 5
Metrics:
  rmse: 0.2970442628930023 - 1561568638688
Tags:
  mlflow.runName: myRun
  mlflow.source.name: TrainDecisionTree.scala
Artifacts:
  Artifact 1/5 - level 0
    path: details
    isDir: true
    Artifact 1/1 - level 1
      path: details/treeModel.txt
      isDir: false
      fileSize: 252

```

#### Dump Experiment

```
scala -cp target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  org.andre.mlflow.tools.DumpExperiment \
  --experimentId 2 \
  --artifactMaxLevel 5 \
  --showRunInfo --showRunData
```

```
Experiment Details:
  experimentId: 2
  name: sklearn_wine_elasticNet
  artifactLocation: /opt/mlflow/server/mlruns/2
  lifecycleStage: active
  runsCount: 7
Runs:
  Run 1/7:
    RunInfo:
      runId: 033be9f1f7e7494daba64bde62c2cf83
. . .
```


### Dump Experiment Runs to CSV file

Create a flattened table of an experiment's runs and dump to CSV file.

All info, data.params, data.metrics and data.tags fields will be flattened into one table. In order to prevent name clashes, data fields will be prefixed with:
* \_p\_ - params
* \_m\_ - metrics
* \_t\_ - tags

If `outputCsvFile` is not specified, the CSV file will be created from the experiment ID as in `exp_runs_2.csv`.

Since run data (params, metrics, tags) fields are not required to be the same for each run, we build a sparse table. Note the blank values for `_m_rmse` and `_t_exp_id` in the output sample below.

```
scala -cp target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  org.andre.mlflow.tools.RunsToCsvConverter \
  --experimentId 2 \
  --outputCsvFile runs.csv
```
Formatted output sample - with subset of columns for readability.
```
_m_rmse            _p_alpha _t_exp_id _t_mlflow.user endTime       runId                            startTime     
0.7504340478812798 0.001    2         andre          1561673524523 3ec72be101054b5d9caa87feba2d3f20 1561673523591 
0.7504340478812798 0.001    2         andre          1561673429978 831a89ee12894e379518841783b18090 1561673427962 
                   0.5                andre          1561670127154 ddaaab3337fd472ea0dfc071ffda9e72 1561670112506 
                   0.5                andre          1561669962054 223b6bb0a8ca405bba96cd083ac8d584 1561669945008 
0.6793073338113734 0.01     2         andre          1561227895063 867390ad87b14dea9841829a7130c2ea 1561227891052 
0.6793073338113734 0.01     2         andre          1561227887437 b9976197bca74e059a1c8d2c35748d6f 1561227883234 
0.7504340478812797 0.001    2         andre          1561227881226 e68d48bd41914cac857399caeede2a0a 1561227880485 
```

### Search Runs

Executes `search runs` feature.
Documentation: 
[Java](https://mlflow.org/docs/latest/java_api/org/mlflow/tracking/MlflowClient.html#searchRuns-java.util.List-java.lang.String-),
[Python](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_runs)
and [REST](https://mlflow.org/docs/latest/rest-api.html#search-runs).

```
scala -cp target/mlflow-scala-examples-1.0-SNAPSHOT.jar \
  org.andre.mlflow.tools.SearchRuns \
  --experimentId 2 \
  --filter "metrics.rmse < 0.7"
```
```
Found 2 matches
RunInfo:
  runId: 867390ad87b14dea9841829a7130c2ea
  experimentId: 2
  lifecycleStage: active
  userId: andre
  status: FINISHED
  artifactUri: /opt/mlflow/server/mlruns/2/867390ad87b14dea9841829a7130c2ea/artifacts
  startTime: 1561227891052
  endTime:   1561227895063
  startTime: 2019-06-22 14:24:51
  endTime:   2019-06-22 14:24:55
  _duration: 4.011 seconds
RunInfo:
  runId: b9976197bca74e059a1c8d2c35748d6f
  experimentId: 2
. . .

```

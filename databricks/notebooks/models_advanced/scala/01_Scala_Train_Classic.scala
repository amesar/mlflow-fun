// Databricks notebook source
// MAGIC %md ## Train Scala Model using MLflow
// MAGIC 
// MAGIC * Libraries:
// MAGIC   * PyPI package: mlflow 
// MAGIC   * Maven coordinates:
// MAGIC     * org.mlflow:mlflow-client:1.1.0 (latest)
// MAGIC     * ml.combust.mleap:mleap-spark_2.11:0.13.0
// MAGIC * Synopsis:
// MAGIC   * Classifier: DecisionTreeRegressor
// MAGIC   * Logs model as SparkML in MLeap formats
// MAGIC     * Log model directory as an artifact. This is a work-around for the Java client's lack of logModel()
// MAGIC   * Includes: AppUtils and MLeapUtils
// MAGIC * Prediction: [Predict_Scala_Model](https://demo.cloud.databricks.com/#notebook/2387379)

// COMMAND ----------

// MAGIC %md ### Setup

// COMMAND ----------

// MAGIC %run ./AppUtils

// COMMAND ----------

// MAGIC %run ./MLflowUtils

// COMMAND ----------

// MAGIC %md #### Create widgets

// COMMAND ----------

val defaultExperimentName = MlflowUtils.getHomeDir + "/scala_DecisionTree"

dbutils.widgets.text("Experiment", defaultExperimentName)
dbutils.widgets.text("maxDepth", "2")
dbutils.widgets.text("maxBins", "32")

val experimentName = dbutils.widgets.get("Experiment")
val maxDepth = dbutils.widgets.get("maxDepth").toInt
val maxBins = dbutils.widgets.get("maxBins").toInt

// COMMAND ----------

// MAGIC %md #### Globals

// COMMAND ----------

val modelDir = "dbfs:/tmp/mlflow_model_Train_Scala_Model" 
val seed = 2019

// COMMAND ----------

// MAGIC %md #### MLflow Setup

// COMMAND ----------

import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus
import java.io.{File,PrintWriter} 

val client = new MlflowClient()

// COMMAND ----------

val experimentId = MlflowUtils.getOrCreateExperimentId(client, experimentName)
println("Experiment name: "+experimentName)
println("Experiment ID: "+experimentId)

// COMMAND ----------

// MAGIC %md #### Get Data

// COMMAND ----------

val data = readWineData()

// COMMAND ----------

// MAGIC %md ### Train

// COMMAND ----------

// MAGIC %md #### Define trainer

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{DecisionTreeRegressor,DecisionTreeRegressionModel}

// COMMAND ----------

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3),seed)

// COMMAND ----------

import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{DecisionTreeRegressor,DecisionTreeRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.mlflow.api.proto.Service.RunStatus

// COMMAND ----------

val metrics = Seq("rmse","r2", "mae")

// Create MLflow run
val runInfo = client.createRun(experimentId)
val runId = runInfo.getRunId()
println(s"Run ID: $runId")

// Log MLflow parameters
client.logParam(runId, "maxDepth",""+maxDepth)
client.logParam(runId, "maxBins",""+maxBins)
println("Parameters:")
println(s"  maxDepth: $maxDepth")
println(s"  maxBins: $maxBins")

// Create model
val dt = new DecisionTreeRegressor()
  .setLabelCol(colLabel)
  .setFeaturesCol(colFeatures)
  .setMaxBins(maxBins)
  .setMaxDepth(maxDepth)

// Create pipeline
val columns = data.columns.toList.filter(_ != colLabel)
val assembler = new VectorAssembler()
  .setInputCols(columns.toArray)
  .setOutputCol(colFeatures)
val pipeline = new Pipeline().setStages(Array(assembler,dt))

// Fit model
val model = pipeline.fit(trainingData)

// Log MLflow training metrics
val predictions = model.transform(testData)
println("Metrics:")
for (metric <- metrics) { 
  val evaluator = new RegressionEvaluator()
    .setLabelCol(colLabel)
    .setPredictionCol(colPrediction)
    .setMetricName(metric)
  val v = evaluator.evaluate(predictions)
  println(s"  $metric: $v - isLargerBetter: ${evaluator.isLargerBetter}")
  client.logMetric(runId, metric, v)
} 

// MLflow - Save a file artifact
val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
new PrintWriter("tree_model.txt") { write("Learned regression tree model - "+new java.util.Date()+s"\n${treeModel.toDebugString}") ; close }
client.logArtifact(runId, new File("tree_model.txt"), "info")

// MLflow - save models
dbutils.fs.rm(modelDir,true)
saveModelAsSparkMLClassic(client, runId, modelDir, model)
saveModelAsMLeapClassic(client, runId, modelDir, model, predictions) 

// Wrap up MLflow run
client.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())

// COMMAND ----------

// MAGIC %md #### Display Run URI

// COMMAND ----------

MlflowUtils.displayRunUri(experimentId,runId)

// COMMAND ----------

// MAGIC %md #### Return result

// COMMAND ----------

dbutils.notebook.exit(runId+" "+experimentId)
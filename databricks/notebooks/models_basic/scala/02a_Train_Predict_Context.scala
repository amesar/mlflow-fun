// Databricks notebook source
// MAGIC %md # Simple Scala MLflow train and predict notebook
// MAGIC Uses new 1.1.0 MLflowContext

// COMMAND ----------

// MAGIC %md ### Setup

// COMMAND ----------

// MAGIC %run ./common

// COMMAND ----------

import java.nio.file.{Paths,Files}
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{DecisionTreeRegressor,DecisionTreeRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.mlflow.tracking.{MlflowClient,MlflowContext}

// COMMAND ----------

dbutils.widgets.text("maxDepth", "2")
dbutils.widgets.text("maxBins", "32")
val maxDepth = dbutils.widgets.get("maxDepth").toInt
val maxBins = dbutils.widgets.get("maxBins").toInt

// COMMAND ----------

// MAGIC %md ### Prepare Data

// COMMAND ----------

downloadData()

// COMMAND ----------

val data = spark.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load(dataPath.replace("/dbfs","dbfs:"))
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), 2019)

// COMMAND ----------

// MAGIC %md ### MLflow Setup

// COMMAND ----------

val client = new MlflowClient()

// COMMAND ----------

val experimentName = createExperimentName()
val experimentId = getOrCreateExperimentId(client, experimentName)

// COMMAND ----------

val mlflow = new MlflowContext()
mlflow.setExperimentName(experimentName)

// COMMAND ----------

// MAGIC %md ### Train

// COMMAND ----------

val colLabel = "quality"
val colPrediction = "prediction"
val colFeatures = "features"
val metrics = Seq("rmse","r2", "mae")

// COMMAND ----------

// Create MLflow run
val run = mlflow.startRun()

val runInfo = client.createRun(experimentId)
val runId = run.getId()
println(s"Run ID: $runId")

// Log MLflow parameters
run.logParam("maxDepth",""+maxDepth)
run.logParam("maxBins",""+maxBins)
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
  run.logMetric(metric, v)
} 

val (user, notebook) = getUserAndNotebook()
val modelPathDbfs = s"dbfs:/tmp/$user/${notebook}.spark-model"
val modelPathFuse = modelPathDbfs.replace("dbfs:","/dbfs")
model.write.overwrite().save(modelPathDbfs)
run.logArtifacts(Paths.get(modelPathFuse),"spark-model")

// End MLflow run
run.endRun()

// COMMAND ----------

displayRunUri(experimentId,runId)

// COMMAND ----------

// MAGIC %md ### Predict

// COMMAND ----------

val modelPath = client.downloadArtifacts(runId,"spark-model")
  .getAbsolutePath
  .replace("/dbfs","dbfs:")
val model = PipelineModel.load(modelPath)
val predictions = model.transform(data)
val df = predictions.select(colPrediction, colLabel, colFeatures)
display(df)
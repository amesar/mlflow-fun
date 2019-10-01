// Databricks notebook source
// MAGIC %md ## Predict Scala Model 
// MAGIC * Runs predictions for a run from [Train_Scala_Model](https://demo.cloud.databricks.com/#notebook/2387348).
// MAGIC * Reads flavors: Spark ML and MLeap.
// MAGIC * Github file version: 
// MAGIC   * https://github.com/amesar/mlflow-fun/blob/master/examples/scala/src/main/scala/org/andre/mlflow/examples/wine/PredictByRunId.scala

// COMMAND ----------

// MAGIC %run ./AppUtils

// COMMAND ----------

// MAGIC %run ./MLflowUtils

// COMMAND ----------

val defaultRunId = "1810101e15e74123b755de3cf1bc2e6d"
dbutils.widgets.text("Run ID",defaultRunId)

val RUN_MODE = " Run Mode" ; val LAST_RUN = "Last Run"  ; val RUN_ID = "Run ID"
dbutils.widgets.dropdown(RUN_MODE, LAST_RUN,Seq(LAST_RUN,RUN_ID))

val EXPERIMENT = "Experiment"
val defaultExperimentId = "2387370"
dbutils.widgets.text(EXPERIMENT,defaultExperimentId)

val _runId = dbutils.widgets.get("Run ID")
val runMode = dbutils.widgets.get(RUN_MODE)
val experimentId = dbutils.widgets.get(EXPERIMENT)

// COMMAND ----------

import org.mlflow.tracking.MlflowClient
val mlflowClient = new MlflowClient()
val runId = if (runMode == LAST_RUN) MlflowUtils.getLastRunId(mlflowClient,experimentId) else _runId

// COMMAND ----------

MlflowUtils.displayRunUri(experimentId,runId)

// COMMAND ----------

// MAGIC %md ##### Read data

// COMMAND ----------

val data = readWineData()

// COMMAND ----------

val runInfo = mlflowClient.getRun(runId).getInfo
val uri = runInfo.getArtifactUri

// COMMAND ----------

// MAGIC %md #### Define Prediction methods

// COMMAND ----------

// MAGIC %run ./MLeapUtils

// COMMAND ----------

import org.mlflow.tracking.MlflowClient
val client = new MlflowClient()

// COMMAND ----------

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Transformer,PipelineModel}

def showPredictions(model: Transformer, data: DataFrame) {
  val predictions = model.transform(data)
  val df = predictions.select(colPrediction, colLabel, colFeatures)
  display(df)
}

def predictSparkML(runId: String, data: DataFrame) {
  val modelPath = client.downloadArtifacts(runId,"spark-model")
    .getAbsolutePath
    .replace("/dbfs","dbfs:")
  val model = PipelineModel.load(modelPath)
  showPredictions(model, data)
}

def predictMLeap(uri: String, data: DataFrame) {
  val modelPath = "file:" +
    client.downloadArtifacts(runId,"mleap-model/mleap/model")
    .getAbsolutePath
  val model = MLeapUtils.readModel(modelPath)
  showPredictions(model, data)
}

// COMMAND ----------

// MAGIC %md #### Predict using Spark ML format

// COMMAND ----------

predictSparkML(runInfo.getRunId, data)

// COMMAND ----------

// MAGIC %md #### Predict using MLeap format

// COMMAND ----------

predictMLeap(runInfo.getRunId, data)

// COMMAND ----------

// MAGIC %md #### Return result

// COMMAND ----------

dbutils.notebook.exit(runId)
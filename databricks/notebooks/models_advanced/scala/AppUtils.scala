// Databricks notebook source
// MAGIC %md ### Columns in common

// COMMAND ----------

val colLabel = "quality"
val colPrediction = "prediction"
val colFeatures = "features"

// COMMAND ----------

// MAGIC %md ### Data utilities

// COMMAND ----------

import sys.process._

def downloadFile(uri: String, filename: String) = {
  val file = new java.io.File(filename)
  if (file.exists) {
      println(s"File $filename already exists")
  } else {
      println(s"Downloading $uri to $filename")
      new java.net.URL(uri) #> file !!
  }
}

// COMMAND ----------

def downloadWineFile() = {
  val dataPath = "/dbfs/tmp/mlflow_wine-quality3.csv"
  val dataUri = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
  downloadFile(dataUri, dataPath)
  dataPath
}

// COMMAND ----------

def readData(dataPath: String) = {
  spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(dataPath.replace("/dbfs","dbfs:"))
}

// COMMAND ----------

def readWineData() = {
  val dataPath = downloadWineFile()
  readData(dataPath)
}

// COMMAND ----------

// MAGIC %md ### Log Model Methods 
// MAGIC * MLflow - Log model directory as an artifact. This is a work-around for the Java client's lack of logModel()
// MAGIC * Tad bit of confusion between Spark ML DBFS paths and MLflow and MLeap which expect Fuse paths

// COMMAND ----------

// MAGIC %md #### Log model as Spark ML Format

// COMMAND ----------

// MAGIC %run ./MLflowUtils

// COMMAND ----------

import org.apache.spark.ml.PipelineModel
import org.mlflow.tracking.MlflowClient

def saveModelAsSparkMLClassic(client: MlflowClient, runId: String, baseModelDir: String, model: PipelineModel) = {
    val modelDir = s"$baseModelDir/spark-model"
    model.write.overwrite().save(modelDir) // hangs if we pass a Fuse path
    client.logArtifacts(runId, new java.io.File(MlflowUtils.mkFusePath(modelDir)), "spark-model")
}

// COMMAND ----------

import org.mlflow.tracking.{MlflowClient,ActiveRun}
import java.nio.file.{Paths,Files}
  
def saveModelAsSparkMLContext(run: ActiveRun, baseModelDir: String, model: PipelineModel) = {
    val modelDir = s"$baseModelDir/spark-model"
    model.write.overwrite().save(modelDir)
    run.logArtifacts(Paths.get(MlflowUtils.mkFusePath(modelDir)), "spark-model")
}

// COMMAND ----------

// MAGIC %md #### Log model as Mleap Format

// COMMAND ----------

// MAGIC %run ./MLeapUtils

// COMMAND ----------

import org.apache.spark.sql.DataFrame

def saveModelAsMLeapClassic(client: MlflowClient, runId: String, baseModelDir: String, model: PipelineModel, predictions: DataFrame) = {
    val modelDir = new java.io.File(s"${MlflowUtils.mkFusePath(baseModelDir)}/mleap-model")
    modelDir.mkdir
    MLeapUtils.saveModel(model, predictions, "file:"+modelDir.getAbsolutePath)
    client.logArtifacts(runId, modelDir, "mleap-model/mleap/model") // NOTE: Make compatible with MLflow Python mlflow.mleap.log_model  
}

// COMMAND ----------

def saveModelAsMLeapContext(run: ActiveRun, baseModelDir: String, model: PipelineModel, predictions: DataFrame) = {
    val modelDir = Paths.get(MlflowUtils.mkFusePath(s"$baseModelDir/mleap-model"))
    Files.createDirectories(modelDir)
    MLeapUtils.saveModel(model, predictions, "file:"+modelDir.toAbsolutePath.toString)
    run.logArtifacts(modelDir, "mleap-model/mleap/model") // Make compatible with MLflow Python mlflow.mleap.log_model
}
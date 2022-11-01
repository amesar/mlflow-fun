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

val user = dbutils.notebook.getContext().tags("user")

val baseDirDbfs = s"dbfs:/tmp/${user}/mlflow_demo"
val baseDirFuse = baseDirDbfs.replace("dbfs:","/dbfs")

val tmpDirDbfs = s"${baseDirDbfs}/scala_advanced"
val tmpDirFuse = tmpDirDbfs.replace("dbfs:","/dbfs")
dbutils.fs.mkdirs(tmpDirDbfs)

val notebook = dbutils.notebook.getContext.notebookPath.get.split("/").last
val modelScratchDirFuse = s"${tmpDirFuse}/${notebook}/spark-model"
val modelScratchDirDbfs = modelScratchDirFuse.replace("/dbfs","dbfs:")

// COMMAND ----------

val dataPath = s"${baseDirFuse}/wine-quality.csv"
val dataUri = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"

def downloadWineFile() = {
  downloadFile(dataUri, dataPath)
  dataPath
}

// COMMAND ----------

def readCsvData(dataPath: String) = {
  spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(dataPath.replace("/dbfs","dbfs:"))
}

// COMMAND ----------

def readWineData(dataType: String = "csv") = {
  if (dataType == "csv") {
    val dataPath = downloadWineFile()
    println(s"Loading $dataPath")
    (readCsvData(dataPath),dataPath)
  } else if (dataType == "parquet") {
    val dataPath = "dbfs:/mnt/training/wine.parquet"
    println(s"Loading $dataPath")
    (spark.read.format("parquet").load(dataPath),dataPath)
  } else {
    throw new Exception("Input data format not supported: $dataPath")
  }
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

def saveModelAsSparkMLClassic(client: MlflowClient, runId: String, baseModelDirDbfs: String, model: PipelineModel) = {
    val modelDir = s"$baseModelDirDbfs/spark-model"
    model.write.overwrite().save(modelDir) // hangs if we pass a Fuse path
    client.logArtifacts(runId, new java.io.File(MlflowUtils.mkFusePath(modelDir)), "spark-model")
}

// COMMAND ----------

import org.mlflow.tracking.{MlflowClient,ActiveRun}
import java.nio.file.{Paths,Files}
  
def saveModelAsSparkMLContext(run: ActiveRun, baseModelDirDbfs: String, model: PipelineModel) = {
    val modelDir = s"$baseModelDirDbfs/spark-model"
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
    MLeapUtils.saveModelAsSparkBundle("file:"+modelDir.getAbsolutePath, model, predictions)
    client.logArtifacts(runId, modelDir, "mleap-model/mleap/model") // NOTE: Make compatible with MLflow Python mlflow.mleap.log_model  
}

// COMMAND ----------

def saveModelAsMLeapContext(run: ActiveRun, baseModelDir: String, model: PipelineModel, predictions: DataFrame) = {
    val modelDir = Paths.get(MlflowUtils.mkFusePath(s"$baseModelDir/mleap-model"))
    Files.createDirectories(modelDir)
    MLeapUtils.saveModelAsSparkBundle("file:"+modelDir.toAbsolutePath.toString, model, predictions)
    run.logArtifacts(modelDir, "mleap-model/mleap/model") // Make compatible with MLflow Python mlflow.mleap.log_model
}
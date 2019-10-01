// Databricks notebook source
// MAGIC %md ## Simple Scala Hello World
// MAGIC 
// MAGIC Uses new MLflow 1.1.0 [MLflowContext](https://mlflow.org/docs/latest/java_api/org/mlflow/tracking/MlflowContext.html) API.
// MAGIC 
// MAGIC Libraries
// MAGIC * Maven coordinates: org.mlflow:mlflow-client:1.2.0
// MAGIC * PyPI package: mlflow==1.2.0

// COMMAND ----------

// MAGIC %run ./common

// COMMAND ----------

// MAGIC %md #### Create MLflow Java Client

// COMMAND ----------

// DBTITLE 0,Create MLflow Java Client
import java.nio.file.{Paths,Files}
import org.mlflow.tracking.{MlflowClient,MlflowContext}
val client = new MlflowClient()

// COMMAND ----------

// MAGIC %md #### Set experiment

// COMMAND ----------

val experimentName = createExperimentName()
val experimentId = getOrCreateExperimentId(client, experimentName)

// COMMAND ----------

val mlflow = new MlflowContext()
mlflow.setExperimentName(experimentName) // Must exist

// COMMAND ----------

// MAGIC %md #### Create run

// COMMAND ----------

val run = mlflow.startRun("HelloWorldContext Run")
println("Run ID: "+run.getId)

// COMMAND ----------

// MAGIC %md #### Log parameters, metrics and tags

// COMMAND ----------

run.logParam("alpha", ".01")
run.logMetric("rmse", 0.876)
run.setTag("algorithm", "None")

// COMMAND ----------

// MAGIC %md #### Set tags that populate UI fields
// MAGIC See: https://github.com/mlflow/mlflow/blob/master/mlflow/utils/mlflow_tags.py#L7

// COMMAND ----------

val sourceName = dbutils.notebook.getContext.notebookPath.get
run.setTag("mlflow.source.name",sourceName) // populates "Source" field in UI

// COMMAND ----------

// MAGIC %md #### Log artifacts - file and directory

// COMMAND ----------

// Log file artifact
val file = Paths.get("info.txt")
Files.write(file, s"File artifact: ${new java.util.Date()}".getBytes)
run.logArtifact(file)

// Log directory artifact
val dir = Paths.get("/tmp/models")
Files.createDirectories(dir)
Files.write(Paths.get(dir.toString, "model.txt"), s"Directory artifact: ${new java.util.Date()}".getBytes)
run.logArtifacts(dir, "model")

// COMMAND ----------

// MAGIC %md #### End the run

// COMMAND ----------

run.endRun()

// COMMAND ----------

// MAGIC %md #### Display Run

// COMMAND ----------

displayRunUri(experimentId,run.getId)
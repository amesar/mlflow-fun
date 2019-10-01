// Databricks notebook source
// MAGIC %md ## Simple Scala Hello World
// MAGIC 
// MAGIC Uses original classic API.
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
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus
import java.io.{File,PrintWriter}

val client = new MlflowClient()

// COMMAND ----------

// MAGIC %md #### Set up experiment

// COMMAND ----------

val experimentName = createExperimentName()
val experimentId = getOrCreateExperimentId(client, experimentName)

// COMMAND ----------

// MAGIC %md #### Create run

// COMMAND ----------

val runInfo = client.createRun(experimentId)
val runId = runInfo.getRunId()
print("runId: "+runId)

// COMMAND ----------

// MAGIC %md #### Log parameters, metrics and tags

// COMMAND ----------

client.logParam(runId, "alpha", ".01")
client.logMetric(runId, "rmse", 0.876)
client.setTag(runId, "algorithm", "None")

// COMMAND ----------

// MAGIC %md #### Set tags that populate UI fields
// MAGIC See: https://github.com/mlflow/mlflow/blob/master/mlflow/utils/mlflow_tags.py#L7

// COMMAND ----------

val sourceName = dbutils.notebook.getContext.notebookPath.get
client.setTag(runId, "mlflow.source.name",sourceName) // populates "Source" field in UI
client.setTag(runId, "mlflow.runName","myRun") // populates "Run Name" field in UI

// COMMAND ----------

// MAGIC %md #### Log artifacts - file and directory

// COMMAND ----------

// Log file artifact
new PrintWriter("/tmp/info.txt") { write("Run at: "+new java.util.Date()) ; close }
client.logArtifact(runId, new File("/tmp/info.txt"),"info")

// Log directory artifact
val dir = new File("tmp")
dir.mkdir
new PrintWriter(new File(dir, "model.txt")) { write("My model: "+new java.util.Date()) ; close }
client.logArtifacts(runId, dir, "model")

// COMMAND ----------

// MAGIC %md #### Close the run

// COMMAND ----------

client.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())

// COMMAND ----------

// MAGIC %md #### Display Run

// COMMAND ----------

displayRunUri(experimentId,runId)
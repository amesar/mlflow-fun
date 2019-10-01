// Databricks notebook source
import org.mlflow.tracking.MlflowClient

// COMMAND ----------

// Return the ID of an experiment - create it if it doesn't exist

def getOrCreateExperimentId(client: MlflowClient, experimentName: String) = {
  try { 
    client.createExperiment(experimentName)
  } catch { 
    case e: org.mlflow.tracking.MlflowHttpException => { // statusCode 400
      client.getExperimentByName(experimentName).get.getExperimentId
    } 
  } 
} 

// COMMAND ----------

//  Display the URI of the run in the MLflow UI

def displayRunUri(experimentId: String, runId: String) = {
  val hostName = dbutils.notebook.getContext().tags.get("browserHostName").get
  val uri = s"https://$hostName/#mlflow/experiments/$experimentId/runs/$runId"
  displayHTML(s"""<b>Run URI:</b> <a href="$uri">$uri</a>""")
}

// COMMAND ----------

def getUserAndNotebook() = {
  val user = dbutils.notebook.getContext().tags("user")
  val notebook = dbutils.notebook.getContext.notebookPath.get.split("/").last
  (user,notebook)
}

// COMMAND ----------

// Create an experiment in user's home directory based on notebook name

def createExperimentName() = {
  val (user, notebook) = getUserAndNotebook()
  val experimentName = s"/Users/$user/gls_scala_$notebook"
  experimentName
}

// COMMAND ----------

// Download uri if it doesn't exist as file

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

val dataPath = "/dbfs/tmp/mlflow_wine-quality3.csv"
val dataUri = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"

def downloadData() {
  downloadFile(dataUri,dataPath)
}
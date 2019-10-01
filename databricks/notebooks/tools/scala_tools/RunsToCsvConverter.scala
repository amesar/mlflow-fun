// Databricks notebook source
// MAGIC %md # Dumps experiment's runs to CSV file
// MAGIC Following is demonstrated:
// MAGIC * Write runs to CSV file.
// MAGIC * Convert runs to a list of maps representing the runs.

// COMMAND ----------

// MAGIC %run ./RunsToCsvConverter.scala

// COMMAND ----------

// MAGIC %run ./MLflowUtils.scala

// COMMAND ----------

// MAGIC %md ### Setup

// COMMAND ----------

//dbutils.widgets.remove(" Experiment ID or Name")

// COMMAND ----------

dbutils.widgets.text("Experiment ID or Name", "")
val user = dbutils.notebook.getContext().tags("user").split("@")(0).replace(".","_")
dbutils.widgets.text("Output CSV file", s"/dbfs/tmp/runs_${user}.csv")

val experimentIdOrName = dbutils.widgets.get("Experiment ID or Name")

val csvFile = dbutils.widgets.get("Output CSV file")
val csvFileDbfs = csvFile.replace("/dbfs","dbfs:")
val csvFileFuse = csvFile.replace("dbfs:","/dbfs")

// COMMAND ----------

import org.mlflow.tracking.MlflowClient
val client = new MlflowClient()
val exp = MLflowUtils.getExperiment(client, experimentIdOrName)
val experimentId = exp.getExperimentId

// COMMAND ----------

// MAGIC %md ### Write as CSV file

// COMMAND ----------

RunsToCsvConverter.writeToCsvFile(client, experimentId, csvFileFuse)

// COMMAND ----------

dbutils.fs.ls(csvFileDbfs).head

// COMMAND ----------

// MAGIC %md #### Read CSV file in as Spark DataFrame

// COMMAND ----------

val df = spark.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load(csvFileDbfs)
display(df)

// COMMAND ----------

// MAGIC %md ### Convert runs to a list of maps

// COMMAND ----------

val runs = RunsToCsvConverter.convertRunsToMap(client, experimentId)

// COMMAND ----------

// MAGIC %md Show keys

// COMMAND ----------

for ((k,v) <- runs.head) {
  println(s"$k: value=$v")
}

// COMMAND ----------

// MAGIC %md Show run IDs

// COMMAND ----------

println(s"Runs: ${runs.size}")
for ((run,j) <- runs.zipWithIndex) {
  println(s"$j - run_id: ${run("runId")} artifactUri: ${run("artifactUri")}")
}
// Databricks notebook source
// MAGIC %run ./DumpRun.scala

// COMMAND ----------

dbutils.widgets.text(" Run ID", "")
val runId = dbutils.widgets.get(" Run ID")

dbutils.widgets.text("Artifact Max Level", "1")
val artifactMaxLevel = dbutils.widgets.get("Artifact Max Level").toInt

// COMMAND ----------

import org.mlflow.tracking.MlflowClient
DumpRun.dumpRun(new MlflowClient(), runId, artifactMaxLevel)
// Databricks notebook source
// MAGIC %run ./DumpExperiment.scala

// COMMAND ----------

dbutils.widgets.text(" Experiment ID", "")
dbutils.widgets.text("Artifact Max Level", "1")
dbutils.widgets.dropdown("Show run info","no",Seq("yes","no"))
dbutils.widgets.dropdown("Show run data","no",Seq("yes","no"))
dbutils.widgets.dropdown("In Databricks","no",Seq("yes","no"))

val experimentId = dbutils.widgets.get(" Experiment ID")
val artifactMaxLevel = dbutils.widgets.get("Artifact Max Level").toInt
val showInfo = dbutils.widgets.get("Show run info") == "yes"
val showData = dbutils.widgets.get("Show run data") == "yes"
val inDatabricks = dbutils.widgets.get("In Databricks") == "yes"

// COMMAND ----------

import org.mlflow.tracking.MlflowClient
DumpExperiment.dumpExperiment(new MlflowClient(), experimentId, artifactMaxLevel, showInfo, showData, inDatabricks)

// COMMAND ----------

import org.mlflow.tracking.MlflowClient
DumpExperiment.dumpExperiment(new MlflowClient(), experimentId, artifactMaxLevel, showInfo, showData, !inDatabricks)

// COMMAND ----------

dumpExperiment2(new MlflowClient(), experimentId, artifactMaxLevel, showInfo, showData)
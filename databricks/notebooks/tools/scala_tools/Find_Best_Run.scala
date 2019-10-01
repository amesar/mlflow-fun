// Databricks notebook source
// MAGIC %md ### Find Best Run of an experiment
// MAGIC 
// MAGIC * Finds the best run of an experiment by searching for the min or max of a metric.
// MAGIC * Note: Scala version is not up to par with Python version.

// COMMAND ----------

// MAGIC %run ./BestRunUtils

// COMMAND ----------

val defaultExperiment_id = "2630330" // /Shared/experiments/demo/sk_WineQuality_demo
val defaultMetric = "rmse"

// COMMAND ----------

dbutils.widgets.text("Experiment ID", defaultExperiment_id)
dbutils.widgets.text("Metric", defaultMetric)
dbutils.widgets.dropdown("isLargerBetter","no",Seq("yes","no"))

val experimentId = dbutils.widgets.get("Experiment ID")
val metric = dbutils.widgets.get("Metric")
val isLargerBetter = dbutils.widgets.get("isLargerBetter") == "yes"

// COMMAND ----------

import org.mlflow.tracking.MlflowClient
val client = new MlflowClient()

// COMMAND ----------

val best = BestRunUtils.getBestRun(client, experimentId,metric,isLargerBetter)

// COMMAND ----------

// MAGIC %run ./DumpRun.scala

// COMMAND ----------

val runId = best.run.getInfo.getRunId

// COMMAND ----------

DumpRun.dumpRun(client, runId, 0)
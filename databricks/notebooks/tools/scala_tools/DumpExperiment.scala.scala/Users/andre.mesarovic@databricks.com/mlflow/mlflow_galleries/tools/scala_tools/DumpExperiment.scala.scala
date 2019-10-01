// Databricks notebook source
// MAGIC %md DumpExperiment.scala
// MAGIC * https://github.com/amesar/mlflow-fun/blob/master/examples/spark-scala/src/main/scala/org/andre/mlflow/tools/DumpExperiment.scala
// MAGIC * synchronized 2019-06-28

// COMMAND ----------

// MAGIC %run ./DumpRun.scala

// COMMAND ----------

//package org.andre.mlflow.tools
  
import org.mlflow.tracking.MlflowClient

object DumpExperiment {
  def dumpExperiment(client: MlflowClient, experimentId: String, artifactMaxLevel: Int, showRunInfo: Boolean, showRunData: Boolean, inDatabricks: Boolean) {
    val expResponse = client.getExperiment(experimentId)
    val exp1 = expResponse.getExperiment
    println(s"Experiment Details:")
    println(s"  experimentId: ${exp1.getExperimentId}")
    println(s"  name: ${exp1.getName}")
    println(s"  artifactLocation: ${exp1.getArtifactLocation}")
    println(s"  lifecycleStage: ${exp1.getLifecycleStage}")
    println(s"  runsCount: ${expResponse.getRunsCount}")

    println(s"Runs:")
    if (showRunInfo || showRunData) {
        val infos = if (inDatabricks) client.listRunInfos(experimentId) else expResponse.getRunsList
        for((info,j) <- infos.zipWithIndex) {
          println(s"  Run ${j+1}/${infos.size}:")
          if (showRunData) {
            DumpRun.dumpRun(client, info.getRunId, artifactMaxLevel,"    ")
          } else {
            DumpRun.dumpRunInfo(info,"    ")
          }
        }
    }
  }
}

// COMMAND ----------

def dumpExperiment2(client: MlflowClient, experimentId: String, artifactMaxLevel: Int, showRuns: Boolean) {
  val infos = client.listRunInfos(experimentId) 
  println("#infos:"+infos.size)
  for((info,j) <- infos.zipWithIndex) {
     println(s"  Run $j:")
     DumpRun.dumpRun(client, info.getRunId, artifactMaxLevel,"    ")
  }
}

// COMMAND ----------


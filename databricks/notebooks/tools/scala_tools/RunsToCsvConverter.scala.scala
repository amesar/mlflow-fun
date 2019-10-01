// Databricks notebook source
// MAGIC %md RunsToCsvConverter.scala
// MAGIC * https://github.com/amesar/mlflow-fun/blob/master/examples/scala/src/main/scala/org/andre/mlflow/tools/RunsToCsvConverter.scala
// MAGIC * synchronized 2019-07-01

// COMMAND ----------

//package org.andre.mlflow.tools

import java.io.PrintWriter
import scala.collection.mutable
import scala.collection.immutable
import scala.collection.JavaConversions._
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.{Run,RunInfo}

object RunsToCsvConverter {
  val sparseMissingValue = ""
  val defaultDelimiter = ","

  def writeToCsvFile(client: MlflowClient, experimentId: String, outputCsvFile: String, delimiter: String = defaultDelimiter) {
    val runs = convertRunsToMap(client, experimentId)
    if (runs.size > 0) {
      writeRuns(outputCsvFile, runs, delimiter)
    }
  }

  def convertRunsToMap(client: MlflowClient, experimentId: String) = {
    val infos = client.listRunInfos(experimentId)
    if (infos.size == 0) {
      println(s"WARNING: No runs for experiment $experimentId")
      List.empty
    } else {
      println(s"Found ${infos.size} runs for experiment $experimentId")
      val runs = infos.map(info => {
        val run = client.getRun(info.getRunId)
        convertRunToMap(run)
      })
      toSparseMap(runs.toList)
    }
  }

  def toSparseMap(runs: List[Map[String,Any]]) = {
    // Create canonical set of keys by union-ing all run.keys
    val keys = runs.foldLeft(mutable.Set[String]()) { (keys, run) => { keys ++= run.keySet } }

    // Add each run value to each to its appropriate slot in sparse map. 
    runs.map(run => {
      val map = keys.foldLeft(mutable.Map[String,Any]()) { (map,key) => { map += (key -> run.getOrElse(key,sparseMissingValue)) }}
      immutable.TreeMap(map.toArray:_*)
    })
  }

  def convertRunToMap(run: Run) = {
    val map = mutable.Map[String,Any]()
    convertInfoToMap(map, run.getInfo)
    for (x <- run.getData.getParamsList) {
      map += ("_p_"+x.getKey -> x.getValue)
    }
    for (x <- run.getData.getMetricsList) {
      map += ("_m_"+x.getKey -> x.getValue)
    }
    for (x <- run.getData.getTagsList) {
      map += ("_t_"+x.getKey -> x.getValue)
    }
    map.toMap
  }

  def convertInfoToMap(map: mutable.Map[String,Any], info: RunInfo) {
    map += ("runId" -> info.getRunId)
    map += ("experimentId" -> info.getExperimentId)
    map += ("lifecycleStage" -> info.getLifecycleStage)
    map += ("userId" -> info.getUserId)
    map += ("status" -> info.getStatus)
    map += ("artifactUri" -> info.getArtifactUri)
    map += ("startTime" -> info.getStartTime)
    map += ("endTime" -> info.getEndTime)
  }

  def writeRuns(path: String, runs: List[Map[String,Any]], delimiter: String) {
    new PrintWriter(path) { 
      for((run,j) <- runs.zipWithIndex) {
        if (j == 0) {
          write(run.keys.mkString(delimiter)+"\n")
        }
        write(run.values.mkString(delimiter)+"\n")
      }
      close
    }
  }
}
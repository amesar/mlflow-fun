package org.andre.mlflow.tools

import java.io.PrintWriter
import scala.collection.mutable
import scala.collection.immutable
import scala.collection.JavaConversions._
import com.beust.jcommander.{JCommander, Parameter}
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLflowUtils
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

  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    val outputCsvFile = if (opts.outputCsvFile == null) s"exp_runs_${opts.experimentId}.csv" else opts.outputCsvFile
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  experimentId: ${opts.experimentId}")
    println(s"  outputCsvFile: ${outputCsvFile}")
    println(s"  delimiter: ${opts.delimiter}")
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    writeToCsvFile(client, opts.experimentId, outputCsvFile, opts.delimiter)
  }

  object opts {
    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--experimentId" ), description = "experimentId", required=true)
    var experimentId: String = null

    @Parameter(names = Array("--delimiter" ), description = "CSV delimiter", required=false)
    var delimiter: String = ","

    @Parameter(names = Array("--outputCsvFile" ), description = "Output CSV File", required=false)
    var outputCsvFile: String = null
  }
}

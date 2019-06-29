package org.andre.mlflow.tools

import java.util.Date
import scala.collection.JavaConversions._
import com.beust.jcommander.{JCommander, Parameter}
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLflowUtils
import org.mlflow.api.proto.Service.RunInfo

object DumpRun {
  val FORMAT = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss")

  def dumpRun(client: MlflowClient, runId: String, artifactMaxLevel: Int, indent: String = "") {
    val run = client.getRun(runId)
    dumpRunInfo(run.getInfo, indent)
    println(s"${indent}Params:")
    for (x <- run.getData.getParamsList) {
      println(s"${indent}  ${x.getKey}: ${x.getValue}")
    }
    println(s"${indent}Metrics:")
    for (x <- run.getData.getMetricsList) {
      println(s"${indent}  ${x.getKey}: ${x.getValue} - ${x.getTimestamp}")
    }
    println(s"${indent}Tags:")
    for (x <- run.getData.getTagsList) {
      println(s"${indent}  ${x.getKey}: ${x.getValue}")
    }
    println(s"${indent}Artifacts:")
    dumpArtifacts(client, runId, "", 0, artifactMaxLevel, indent+"  ")
  }

  def dumpRunInfo(info: RunInfo, indent: String = "") {
    println(s"${indent}RunInfo:")
    println(s"$indent  runId: ${info.getRunId}")
    println(s"$indent  experimentId: ${info.getExperimentId}")
    println(s"$indent  lifecycleStage: ${info.getLifecycleStage}")
    println(s"$indent  userId: ${info.getUserId}")
    println(s"$indent  status: ${info.getStatus}")
    println(s"$indent  artifactUri: ${info.getArtifactUri}")
    println(s"$indent  startTime: ${info.getStartTime}")
    println(s"$indent  endTime:   ${info.getEndTime}")
    println(s"$indent  startTime: ${FORMAT.format(new Date(info.getStartTime))}")
    println(s"$indent  endTime:   ${FORMAT.format(new Date(info.getEndTime))}")
    val duration = (info.getEndTime - info.getStartTime).toDouble / 1000
    println(s"$indent  _duration: ${duration} seconds")
  }

  def dumpArtifacts(client: MlflowClient, runId: String, path: String, level: Int, maxLevel: Int, indent: String) {
    if (level+1 > maxLevel) {
       return
    }
    val artifacts = client.listArtifacts(runId, path)
    for ((art,j) <- artifacts.zipWithIndex) {
      println(s"${indent}Artifact ${j+1}/${artifacts.size} - level $level")
      println(s"$indent  path: ${art.getPath}")
      println(s"$indent  isDir: ${art.getIsDir}")
      if (art.getIsDir) {
        dumpArtifacts(client, runId, art.getPath, level+1, maxLevel, indent+"  ")
      } else {
        println(s"$indent  fileSize: ${art.getFileSize}")
      }
    }
  }

  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  runId: ${opts.runId}")
    println(s"  artifactMaxLevel: ${opts.artifactMaxLevel}")
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    dumpRun(client, opts.runId, opts.artifactMaxLevel)
  }

  object opts {
    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--runId" ), description = "runId", required=true)
    var runId: String = null

    @Parameter(names = Array("--artifactMaxLevel" ), description = "Number of artifact levels to recurse", required=false)
    var artifactMaxLevel = 1
  }
}

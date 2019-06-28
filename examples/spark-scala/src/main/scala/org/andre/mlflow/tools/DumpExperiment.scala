package org.andre.mlflow.tools

import scala.collection.JavaConversions._
import com.beust.jcommander.{JCommander, Parameter}
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLflowUtils

object DumpExperiment {

  def dumpExperiment(client: MlflowClient, experimentId: String, artifactMaxLevel: Int, showRuns: Boolean) {
    val exp0 = client.getExperiment(experimentId)
    val exp1 = exp0.getExperiment
    println(s"Experiment Details:")
    println(s"  experimentId: ${exp1.getExperimentId}")
    println(s"  name: ${exp1.getName}")
    println(s"  artifactLocation: ${exp1.getArtifactLocation}")
    println(s"  lifecycleStage: ${exp1.getLifecycleStage}")
    println(s"  runsCount: ${exp0.getRunsCount}")

    println(s"Runs:")
    if (showRuns) {
      for((info,j) <- exp0.getRunsList.zipWithIndex) {
        println(s"  Run $j:")
        DumpRun.dumpRun(client, info.getRunId, artifactMaxLevel,"    ")
      }
    } else {
      for(info <- exp0.getRunsList) {
        DumpRun.dumpRunInfo(info,"  ")
      }
    }
  }

  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  experimentId: ${opts.experimentId}")
    println(s"  artifactMaxLevel: ${opts.artifactMaxLevel}")
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    dumpExperiment(client, opts.experimentId, opts.artifactMaxLevel, opts.showRuns)
  }

  object opts {
    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--experimentId" ), description = "rexperimentId", required=true)
    var experimentId: String = null

    @Parameter(names = Array("--artifactMaxLevel" ), description = "Number of artifact levels to recurse", required=false)
    var artifactMaxLevel = 1

    @Parameter(names = Array("--showRuns" ), description = "Show run details", required=false)
    var showRuns = false
  }
}

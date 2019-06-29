package org.andre.mlflow.tools

import scala.collection.JavaConversions._
import com.beust.jcommander.{JCommander, Parameter}
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLflowUtils

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

  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  experimentId: ${opts.experimentId}")
    println(s"  artifactMaxLevel: ${opts.artifactMaxLevel}")
    println(s"  showRunInfo: ${opts.showRunInfo}")
    println(s"  showRunData: ${opts.showRunData}")
    println(s"  inDatabricks: ${opts.inDatabricks}")
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    dumpExperiment(client, opts.experimentId, opts.artifactMaxLevel, opts.showRunInfo, opts.showRunData, opts.inDatabricks)
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

    @Parameter(names = Array("--showRunInfo" ), description = "Show run info", required=false)
    var showRunInfo = false

    @Parameter(names = Array("--showRunData" ), description = "Show run info and data", required=false)
    var showRunData = false

    @Parameter(names = Array("--inDatabricks" ), description = "Running inside Databricks", required=false)
    var inDatabricks = false
  }
}

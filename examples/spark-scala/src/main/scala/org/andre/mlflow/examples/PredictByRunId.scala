package org.andre.mlflow.examples

import com.beust.jcommander.{JCommander, Parameter}
import org.andre.mlflow.util.MLflowUtils

object PredictByRunId {
  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  runId: ${opts.runId}")

    val mlflowClient = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    val runInfo = mlflowClient.getRun(opts.runId).getInfo
    PredictUtils.predict(runInfo, opts.dataPath)
  }

  object opts {
    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null

    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--runId" ), description = "runId", required=true)
    var runId: String = null
  }
}

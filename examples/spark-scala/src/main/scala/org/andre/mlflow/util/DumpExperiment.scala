package org.andre.mlflow.util

import com.beust.jcommander.{JCommander, Parameter}

object DumpExperiment {

  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  experimentId: ${opts.experimentId}")
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    val exp = client.getExperiment(opts.experimentId)
    println(exp)
  }

  object opts {
    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--experimentId" ), description = "rexperimentId", required=true)
    var experimentId: String = null
  }
}

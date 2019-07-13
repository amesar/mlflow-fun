package org.andre.mlflow.examples.wine

import com.beust.jcommander.{JCommander, Parameter}
import org.andre.mlflow.util.{MLflowUtils,BestRunUtils}

object PredictByBestRun {
  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  experimentId: ${opts.experimentId}")
    println(s"  metric: ${opts.metric}")
    println(s"  ascending: ${opts.ascending}")

    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)

    val best = BestRunUtils.getBestRun(client, opts.experimentId, opts.metric, opts.ascending)
    println(s"best.runId: ${best.run.getInfo.getRunUuid}")
    println(s"best.value: ${best.value}")
    PredictUtils.predict(best.run.getInfo, opts.dataPath)
  }

  object opts {
    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null

    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--experimentId" ), description = "experimentId", required=true)
    var experimentId: String = null

    @Parameter(names = Array("--metric" ), description = "metric", required=true)
    var metric: String = null

    @Parameter(names = Array("--ascending" ), description = "ascending", required=false)
    var ascending = false
  }
}

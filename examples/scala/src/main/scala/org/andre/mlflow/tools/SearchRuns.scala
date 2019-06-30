package org.andre.mlflow.tools

import scala.collection.JavaConversions._
import com.beust.jcommander.{JCommander, Parameter}
import org.andre.mlflow.util.MLflowUtils

object SearchRuns {
  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  experimentId: ${opts.experimentId}")
    println(s"  filter: ${opts.filter}")
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)

    val infos = client.searchRuns(Seq(opts.experimentId),opts.filter)
    println(s"Found ${infos.size} matches")
    for (info <- infos) {
      DumpRun.dumpRunInfo(info)
    }
  }

  object opts {
    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--experimentId" ), description = "experimentId", required=true)
    var experimentId: String = null

    @Parameter(names = Array("--filter" ), description = "filter", required=false)
    var filter: String = ""
  }
}

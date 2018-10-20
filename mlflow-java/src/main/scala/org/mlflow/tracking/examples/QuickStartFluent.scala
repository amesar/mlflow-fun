package org.mlflow.tracking.examples

import java.io.{File,PrintWriter}
import org.mlflow.tracking.{MlflowClient,RunContext}
import org.mlflow.tracking.creds.BasicMlflowHostCreds

object QuickStartFluent {
  def main(args: Array[String]) {
    val trackingUri = args(0)
    println(s"Tracking URI: $trackingUri")

    val mlflowClient = 
      if (args.length > 1) {
        new MlflowClient(new BasicMlflowHostCreds(trackingUri,args(1)))
      } else {
        new MlflowClient(trackingUri)
      }

    val expName = "scala/QuickStart"
    val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)
    println("Experiment name: "+expName)
    println("Experiment ID: "+expId)

    val sourceName = getClass().getSimpleName()+".scala"
    println(s"sourceName: $sourceName")

    new RunContext(mlflowClient, expId, sourceName) {
      println(s"Run ID: ${getRunId()}")

      logParam("param1","p1")
      logMetric("metric1",0.123F)
      setTag("fluent","true")

      new PrintWriter("/tmp/info.txt") { write("Info: "+new java.util.Date()) ; close }
      logArtifact(new File("/tmp/info.txt"),"info")

      val dir = new File("/tmp/run_artifacts")
      dir.mkdirs()
      new PrintWriter(dir+"/info1.txt") { write("Info1 at: "+new java.util.Date()) ; close }
      new PrintWriter(dir+"/info2.txt") { write("Info2 at: "+new java.util.Date()) ; close }
      logArtifacts(dir)
      logArtifacts(dir,"dir")
    }
  }
}

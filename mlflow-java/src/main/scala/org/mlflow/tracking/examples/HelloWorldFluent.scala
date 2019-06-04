package org.mlflow.tracking.examples

import java.io.{File,PrintWriter}
import org.mlflow.tracking.{MlflowClient,RunContext}
import org.mlflow.tracking.creds.BasicMlflowHostCreds
import org.mlflow.util.MLflowUtils

object HelloWorldFluent {
  def main(args: Array[String]) {
    val trackingUri = args(0)
    println(s"Tracking URI: $trackingUri")

    val mlflowClient = 
      if (args.length > 1) {
        new MlflowClient(new BasicMlflowHostCreds(trackingUri,args(1)))
      } else {
        new MlflowClient(trackingUri)
      }

    val expName = "scala/HelloWorld"
    val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)
    println(s"Experiment name: $expName")
    println(s"Experiment ID: $expId")

    val sourceName = getClass().getSimpleName()+".scala"
    println(s"Source Name: $sourceName")

    new RunContext(mlflowClient, expId, sourceName) {
      println(s"Run ID: ${getRunId()}")

      logParam("param1","p1")
      logMetric("metric1",0.123987651)
      setTag("fluent","true")
      setTag("mlflow.note.content","my **bold** note")

      val now = new java.util.Date()
      new PrintWriter("/tmp/info.txt") { write(s"Info: $now") ; close }
      logArtifact(new File("/tmp/info.txt"),"info")

      val dir = new File("/tmp/run_artifacts")
      dir.mkdirs()
      new PrintWriter(s"$dir/info1.txt") { write(s"Info1 at: $now") ; close }
      new PrintWriter(s"$dir/info2.txt") { write(s"Info2 at: $now") ; close }
      logArtifacts(dir)
      logArtifacts(dir,"dir")
    }
  }
}

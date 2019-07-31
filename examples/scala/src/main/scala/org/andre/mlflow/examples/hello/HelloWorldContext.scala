package org.andre.mlflow.examples.hello

import java.io.PrintWriter
import java.nio.file.{Paths,Files}
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.{RunStatus,CreateRun}
import scala.collection.JavaConversions._
import org.andre.mlflow.util.MLflowUtils
import org.mlflow.tracking.MlflowContext


object HelloWorldContext {
  val now = new java.util.Date()

  def main(args: Array[String]) {
    val mlflow = new MlflowContext()
    val client = mlflow.getClient()

    // Create and set experiment
    val expName = "scala_HelloWorld_Context"
    val expId = MLflowUtils.getOrCreateExperimentId(client, expName)
    println("Experiment name: "+expName)
    println("Experiment ID: "+expId)
    mlflow.setExperimentName(expName)

    // Create run
    val run = mlflow.startRun("HelloWorldContext Run")
    println("Run ID: "+run.getId)

    // Log params and metrics
    run.logParam("alpha","0.5")
    run.logMetric("rmse",0.876)
    run.setTag("origin","laptop")
    run.setTag("mlflow.source.name",MLflowUtils.getSourceName(getClass())) // populates "Source" field in UI

    // Log file artifact
    val tdir = "tmp"
    val filePath = Paths.get(tdir,"info.txt")
    Files.write(filePath, s"File artifact: $now".getBytes)
    run.logArtifact(filePath)

    // Log directory artifact
    val dirPath = Paths.get(tdir)
    if (!Files.exists(dirPath)) {
        Files.createDirectory(dirPath)
    }
    val modelPath = Paths.get(tdir,"model.txt")
    Files.write(modelPath, s"My model: $now".getBytes)
    run.logArtifact(modelPath,"model")

    // Bye bye
    run.endRun();
  }
}

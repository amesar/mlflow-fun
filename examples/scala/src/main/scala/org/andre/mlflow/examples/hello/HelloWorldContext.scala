package org.andre.mlflow.examples.hello

import java.nio.file.{Paths,Files}
import org.mlflow.tracking.{MlflowClient,MlflowContext}
import org.andre.mlflow.util.MLflowUtils

/**
 * Scala Hello World using MLflow Java client with MlflowContext.
 */
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
    val odir = Paths.get("tmp")
    Files.createDirectories(odir)
    val file = Paths.get(odir.toString,"info.txt")
    Files.write(file, s"File artifact: $now".getBytes)
    run.logArtifact(file)

    // Log directory artifact - mock model
    val modelPath = Paths.get(odir.toString,"model.txt")
    Files.write(modelPath, s"Directory artifact: $now".getBytes)
    run.logArtifact(modelPath,"model")

    // Close run
    run.endRun()
  }
}

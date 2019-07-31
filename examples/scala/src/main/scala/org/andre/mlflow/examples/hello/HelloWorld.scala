package org.andre.mlflow.examples.hello

import java.io.{File,PrintWriter}
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.{RunStatus,CreateRun}
import scala.collection.JavaConversions._
import org.andre.mlflow.util.MLflowUtils

object HelloWorld {
  def main(args: Array[String]) {

    // Create MLflow client
    val client = MLflowUtils.createMlflowClient(args)

    // Create or get existing experiment
    val expName = "scala_HelloWorld"
    val expId = MLflowUtils.getOrCreateExperimentId(client, expName)
    println("Experiment name: "+expName)
    println("Experiment ID: "+expId)

    // Create run
    val runInfo = client.createRun(expId)
    val runId = runInfo.getRunUuid()
    println("Run ID: "+runId)

    // Log params and metrics
    client.logParam(runId, "alpha","0.5")
    client.logMetric(runId, "rmse",0.876)
    client.setTag(runId, "origin","laptop")
    client.setTag(runId, "mlflow.source.name",MLflowUtils.getSourceName(getClass())) // populates "Source" field in UI
    client.setTag(runId, "mlflow.runName","myRun") // populates "Run Name" field in UI

    // Log file artifact
    new PrintWriter("info.txt") { write("File artifact: "+new java.util.Date()) ; close }
    client.logArtifact(runId, new File("info.txt"))

    // Log directory artifact
    val dir = new File("tmp")
    dir.mkdir
    new PrintWriter(new File(dir, "model.txt")) { write("Directory artifact: "+new java.util.Date()) ; close }
    client.logArtifacts(runId, dir, "model")

    // Close run
    client.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
  }
}

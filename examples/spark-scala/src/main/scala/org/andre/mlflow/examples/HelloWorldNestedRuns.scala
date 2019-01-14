package org.andre.mlflow.examples

import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds.BasicMlflowHostCreds
import org.mlflow.api.proto.Service.RunStatus

object HelloWorldNestedRuns {
  def main(args: Array[String]) {
    val trackingUri = args(0)
    println(s"Tracking URI: $trackingUri")

    // Create client
    val mlflowClient = MLflowUtils.createMlflowClient(args)

    // Create or get existing experiment
    val expName = "scala/QuickStart"
    val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)
    println("Experiment name: "+expName)
    println("Experiment ID: "+expId)

    // Create run
    val sourceName = getClass().getSimpleName()+".scala"
    val runInfo = mlflowClient.createRun(expId, sourceName);
    val runId = runInfo.getRunUuid()

    // Log params and metrics
    mlflowClient.logParam(runId, "p1","hi")
    mlflowClient.logMetric(runId, "m1",0.123F)

    // Close run
    mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
  }
}

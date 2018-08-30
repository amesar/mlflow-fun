package org.andre.mlflow.examples

import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds.BasicMlflowHostCreds
import org.mlflow.api.proto.Service.RunStatus

object QuickStart {
  def main(args: Array[String]): Unit = {
    val trackingUri = args(0)
    println(s"Tracking URI: $trackingUri")

    val mlflowClient = 
      if (args.length > 1) {
        new MlflowClient(new BasicMlflowHostCreds(trackingUri,args(1)))
      } else {
        new MlflowClient(trackingUri)
      }

    // Create or get existing experiment
    val expName = "scala/QuickStart"
    val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)
    println("Experiment name: "+expName)
    println("Experiment ID: "+expId)

    // Create run
    val sourceName = getClass().getSimpleName()+".scala"
    val runInfo = mlflowClient.createRun(expId, sourceName);
    val runId = runInfo.getRunUuid()

    // Log stuff
    mlflowClient.logParam(runId, "p1","hi")
    mlflowClient.logMetric(runId, "m1",0.123F)

    // Close run
    mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
  }
}

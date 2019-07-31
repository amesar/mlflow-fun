package org.andre.mlflow.examples.hello

import java.io.{File,PrintWriter}
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.{RunStatus,CreateRun}
import org.andre.mlflow.util.MLflowUtils

/**
 * Shows how to set run status if failure occurs during run. 
 */
object ResilientHelloWorld {
  def main(args: Array[String]) {
    val client = MLflowUtils.createMlflowClient(args(0))
    val doThrowException = args.size > 1
    println("doThrowException: "+doThrowException)

    val expName = "scala_ResilientHelloWorld"
    val experimentId = MLflowUtils.getOrCreateExperimentId(client, expName)
    println("Experiment name: "+expName)
    println("Experiment ID: "+experimentId)

    val runId = client.createRun(experimentId).getRunId()
    println("Run ID: "+runId)
    try {
      if (doThrowException) {
        throw new Exception("Ouch")
      }
      client.logParam(runId, "alpha", "0.5")
      client.logMetric(runId, "rmse", 0.786)
      client.setTag(runId, "origin", "laptop")
      println("Status OK")
      client.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
    } catch {
      case e: Exception => {
        println("Status FAILED: "+e)
        client.setTerminated(runId, RunStatus.FAILED, System.currentTimeMillis())
      }
    }
    val run = client.getRun(runId)
    println(s"\nRetrieved run: runId: $runId - status: ${run.getInfo.getStatus}")
  }
}

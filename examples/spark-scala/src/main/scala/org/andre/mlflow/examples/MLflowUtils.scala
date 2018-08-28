package org.andre.mlflow.examples

import scala.collection.JavaConversions._
import org.mlflow.tracking.MlflowClient

object MLflowUtils {
  def getOrCreateExperimentId(client: MlflowClient, experimentName: String) : Long = {
    val expOpt = client.listExperiments() find (_.getName == experimentName)
    expOpt match {
      case Some(exp) => exp.getExperimentId
      case None => client.createExperiment(experimentName)
    }
  }
}

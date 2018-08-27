package org.andre.mlflow.examples

import scala.collection.JavaConversions._
import org.mlflow.client.ApiClient

object MLflowUtils {
  def getOrCreateExperimentId(client: ApiClient, experimentName: String) : Long = {
    val expOpt = client.listExperiments() find (_.getName == experimentName)
    expOpt match {
      case Some(exp) => exp.getExperimentId
      case None => client.createExperiment(experimentName)
    }
  }
}

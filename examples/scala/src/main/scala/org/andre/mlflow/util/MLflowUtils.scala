package org.andre.mlflow.util

import scala.collection.JavaConversions._
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds.BasicMlflowHostCreds
import org.mlflow.api.proto.Service.Run

object MLflowUtils {

  def setExperiment(client: MlflowClient, experimentName: String) : String = {
    val expOpt = client.listExperiments() find (_.getName == experimentName)
    expOpt match {
      case Some(exp) => exp.getExperimentId
      case None => client.createExperiment(experimentName)
    }
  }

  def getExperiment(client: MlflowClient, experimentIdOrName: String) = {
    if (isNumeric(experimentIdOrName)) {
      try {
        val expResponse = client.getExperiment(experimentIdOrName)
        expResponse.getExperiment()
      } catch {
        case e: org.mlflow.tracking.MlflowHttpException => {
          throw new NoSuchElementException(s"Cannot find experiment name '$experimentIdOrName'. ${e}")
        }
      }
    } else {
      val expOpt = client.getExperimentByName(experimentIdOrName)
      expOpt.isPresent  match {
        case true => expOpt.get()
        case _ => throw new NoSuchElementException(s"Cannot find experiment name '$experimentIdOrName'")
      }
    }
  }

  def isNumeric(input: String) = input.forall(_.isDigit)

  def createMlflowClient(args: Array[String]) = {
    println("args: "+args)
    if (args.length == 0) {
        val env = System.getenv("MLFLOW_TRACKING_URI")
        println(s"MLFLOW_TRACKING_URI: $env")
        new MlflowClient()
    } else {
      val trackingUri = args(0)
      println(s"Tracking URI: $trackingUri")
      if (args.length > 1) {
        new MlflowClient(new BasicMlflowHostCreds(trackingUri,args(1)))
      } else {
        new MlflowClient(trackingUri)
      }
    }
  }

  def createMlflowClient(trackingUri: String, token: String) = {
    if (trackingUri == null) {
        val env = System.getenv("MLFLOW_TRACKING_URI")
        println(s"MLFLOW_TRACKING_URI: $env")
        new MlflowClient()
    } else {
      if (token != null) {
        new MlflowClient(new BasicMlflowHostCreds(trackingUri, token))
      } else {
        new MlflowClient(trackingUri)
      }
    }
  }

  def getLastRunInfo(client: MlflowClient, experimentId: String) = {
    val infos = client.listRunInfos(experimentId) 
    infos.sortWith(_.getStartTime > _.getStartTime)(0)
  }

  def getLastRun(client: MlflowClient, experimentId: String) = {
    val info = getLastRunInfo(client, experimentId)
    client.getRun(info.getRunUuid) 
  }

  def getSourceName(clazz: Class[_]) = clazz.getSimpleName().replace("$","")+".scala"
}

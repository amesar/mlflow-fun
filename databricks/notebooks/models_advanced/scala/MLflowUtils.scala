// Databricks notebook source
import org.mlflow.tracking.{MlflowClient,MlflowHttpException}
import scala.collection.JavaConversions._

object MlflowUtils {
   
  def getOrCreateExperimentId(client: MlflowClient, experimentName: String) : String = {
    try {
      val experimentId = client.createExperiment(experimentName)
      println(s"Created new experiment: $experimentName")
      experimentId
    } catch { 
      case e: org.mlflow.tracking.MlflowHttpException => { // statusCode 400
        client.getExperimentByName(experimentName).get.getExperimentId
      }
    } 
  } 
  
  def createRunUri(experimentId: String, runId: String) = {
    val opt = dbutils.notebook.getContext().tags.get("browserHostName")
    opt match {
      case Some(hostName) => s"https://${hostName}/#mlflow/experiments/$experimentId/runs/$runId"
      case None => ""
    }
  }
  
  def displayRunUri(experimentId: String, runId: String) = {
    val opt = dbutils.notebook.getContext().tags.get("browserHostName")
    val hostName = opt match {
      case Some(h) => h
      case None => "_?_"
    }
    val uri = s"https://$hostName/#mlflow/experiments/$experimentId/runs/$runId"
    displayHTML(s"""Run URI: <a href="$uri">$uri</a>""")
  }
  
  def getUsername() = dbutils.notebook.getContext().tags("user")

  def getHomeDir() = "/Users/"+getUsername
  
  def getSourceName() = dbutils.notebook.getContext.notebookPath.get
  
  def getNotebookName =  dbutils.notebook.getContext.notebookPath.get.split("/").last
  
  def mkFusePath(dbfsPath: String) =  dbfsPath.replace("dbfs:","/dbfs")
  
  def getLastRunInfo(client: MlflowClient, experimentId: String) = {
    val infos = client.listRunInfos(experimentId)
    infos.sortWith(_.getStartTime > _.getStartTime)(0)
  }
  
  def getLastRunId(client: MlflowClient, experimentId: String) = {
    val infos = client.listRunInfos(experimentId)
    infos.sortWith(_.getStartTime > _.getStartTime)(0).getRunId()
  }

  def getLastRun(client: MlflowClient, experimentId: String) = {
    val info = getLastRunInfo(client, experimentId)
    client.getRun(info.getRunId)
  }
}

// COMMAND ----------

// Source: https://github.com/amesar/mlflow-fun/blob/master/examples/spark-scala/src/main/scala/org/andre/mlflow/util/BestRunUtils.scala
// Date: 2019-06-10

import scala.collection.JavaConversions._
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.Run

object BestRunUtils {
  case class Best(var run: Run, var value: Double)
  def gt(x: Double, y: Double) : Boolean = x>y
  def lt(x: Double, y: Double) : Boolean = x<y

  def getBestRun(client: MlflowClient, experimentId: String, metricName: String, ascending: Boolean=false) = {
    val (init,funk) = if (ascending) {
      (scala.Double.MinValue, lt _)
    } else {
      (scala.Double.MaxValue, gt _)
    }
    val best = Best(null,init)
    val infos = client.listRunInfos(experimentId) 
    for (info <- infos) {
      val run = client.getRun(info.getRunUuid) 
      calcBest(best,run,metricName, funk)
    }
    best
  }

  def calcBest(best: Best, run: Run, metricName: String, funk:(Double, Double) => Boolean ) {
    if (best.run == null) {
      best.run = run
      return
    }
    for (m <- run.getData.getMetricsList) {
      if (metricName == m.getKey) {
        if (funk(best.value,m.getValue)) {
          best.run = run
          best.value = m.getValue
          return
        }
      }
    }
  }
}
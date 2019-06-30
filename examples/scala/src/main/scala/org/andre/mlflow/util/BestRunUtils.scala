package org.andre.mlflow.util

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

// Databricks notebook source
// MAGIC %md MLflowUtils.scala
// MAGIC * https://github.com/amesar/mlflow-fun/blob/master/examples/scala/src/main/scala/org/andre/mlflow/util/MLflowUtils.scala
// MAGIC * synchronized 2019-07-01

// COMMAND ----------

// package org.andre.mlflow.util
import org.mlflow.tracking.MlflowClient

object MLflowUtils {
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
}

// COMMAND ----------


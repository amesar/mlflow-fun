package org.mlflow

import org.testng.Assert._
import org.testng.annotations._
import org.mlflow.tracking.MlflowClient
import scala.util.Properties;

class BaseTest() {
  val trackingUriDefault = "http://localhost:5000";
  var client: MlflowClient = null 

  @BeforeClass
  def BeforeClass() {
      val trackingUri = Properties.envOrElse("MLFLOW_TRACKING_URI",trackingUriDefault)
      println(s"trackingUri=$trackingUri")
      client = new MlflowClient(trackingUri);
  }

  def createExperimentName() = "TestScala_"+System.currentTimeMillis.toString
}

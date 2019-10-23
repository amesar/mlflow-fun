package org.andre.mlflow.util

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import ml.combust.mleap.runtime.MleapSupport._
import resource.managed
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

/*
  MLeap URI formats:
        file:/tmp/mleap_scala_model_export/my-model
    jar:file:/tmp/mleap_scala_model_export/my-model.zip
*/
object MLeapUtils {

  def saveModelAsSparkBundle(bundlePath: String, model: PipelineModel, df: DataFrame) {
    val context = SparkBundleContext().withDataset(df)
    (for(modelFile <- managed(BundleFile(bundlePath))) yield {
      model.writeBundle.save(modelFile)(context)
    }).tried.get
  }

  def readModelAsSparkBundle(bundlePath: String) = {
    val obundle = (for(bundle <- managed(BundleFile(bundlePath))) yield {
      bundle.loadSparkBundle().get
    }).opt
    obundle match {
      case Some(b) => b.root
      case None => throw new Exception(s"Internal MLeap NPE error: $bundlePath")
    }
  }
  
  def readModelAsMLeapBundle(bundlePath: String) = {
    val obundle = (for(bundle <- managed(BundleFile(bundlePath))) yield {
      bundle.loadMleapBundle().get
    }).opt
    obundle match {
      case Some(b) => b.root
      case None => throw new Exception(s"Internal MLeap NPE error: $bundlePath")
    }
  }
  
  import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
  import ml.combust.mleap.core.types._

  def makeLeapFrame(data: DataFrame) = {
    val schema = org.apache.spark.sql.mleap.TypeConverters.sparkSchemaToMleapSchema(data)
    val transformedData = data.limit(1)
      .collect
      .map { sparkRow: org.apache.spark.sql.Row =>
        ml.combust.mleap.runtime.frame.Row(org.apache.spark.sql.Row.unapplySeq(sparkRow).get: _*)
      }
    DefaultLeapFrame(schema, transformedData)
  }
}

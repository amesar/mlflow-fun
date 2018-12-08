package org.andre.mlflow.examples

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
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

  def save(model: PipelineModel, df: DataFrame, bundlePath: String) {
    val context = SparkBundleContext().withDataset(df)
    (for(modelFile <- managed(BundleFile(bundlePath))) yield {
      model.writeBundle.save(modelFile)(context)
    }).tried.get
  }

  def read(bundlePath: String) = {
    val opt = (for(bundle <- managed(BundleFile(bundlePath))) yield {
      bundle.loadSparkBundle().get
    }).opt
    val bundle = opt.get
    val model = bundle.root
    model
  }
}

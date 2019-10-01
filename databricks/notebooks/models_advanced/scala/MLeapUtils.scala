// Databricks notebook source
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.ml.bundle.SparkBundleContext
import resource._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

// COMMAND ----------

object MLeapUtils {
  
  def saveModel(model: PipelineModel, df: DataFrame, bundlePath: String) {
    val context = SparkBundleContext().withDataset(df)
    (for(modelFile <- managed(BundleFile(bundlePath))) yield {
      model.writeBundle.save(modelFile)(context)
    }).tried.get
  }

  def readModel(bundlePath: String) = {
    val obundle = (for(bundle <- managed(BundleFile(bundlePath))) yield {
      bundle.loadSparkBundle().get
    }).opt
    obundle match {
      case Some(b) => b.root
      case None => throw new Exception(s"Cannot find bundle: $bundlePath")
    }
  }
}
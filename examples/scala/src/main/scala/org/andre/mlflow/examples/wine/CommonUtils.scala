package org.andre.mlflow.examples.wine

import org.apache.spark.sql.SparkSession

object CommonUtils {
  val colLabel = "quality"
  val colPrediction = "prediction"
  val colFeatures = "features"

  def readData(spark: SparkSession, dataPath: String) = {
    spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(dataPath)
  }
}

package org.andre.mlflow.examples.libsvm

import org.apache.spark.sql.{SparkSession,DataFrame}
import org.apache.spark.ml.{PipelineModel,Transformer}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.mlflow.api.proto.Service.RunInfo
import org.andre.mlflow.util.MLeapUtils

object PredictUtils {

  def predict(runInfo: RunInfo, dataPath: String) {
    val spark = SparkSession.builder.appName("Predict").getOrCreate()
    val data = spark.read.format("libsvm").load(dataPath)
    val uri = runInfo.getArtifactUri
    predictSparkML(uri, data)
    predictMLeap(uri, data)
  }

  def predictSparkML(uri: String, data: DataFrame) {
    println("==== Spark ML")
    val modelPath = s"${uri}/spark-model"
    val model = PipelineModel.load(modelPath)
    showPredictions(model, data)
  }

  def predictMLeap(uri: String, data: DataFrame) {
    println("==== MLeap")
    val modelPath = s"file:${uri}/mleap-model/mleap/model"
    val model = MLeapUtils.readModelAsSparkBundle(modelPath)
    showPredictions(model, data)
  } 

  def showPredictions(model: Transformer, data: DataFrame) {
    val predictions = model.transform(data)
    val df = predictions.select("features","label","prediction").sort("features","label","prediction")
    df.show(10)
  }

  def evaluatePredictions(predictions: DataFrame) = {
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Metrics:")
    println(s"  RMSE: $rmse")
    println(s"  isLargerBetter: ${evaluator.isLargerBetter}")
    rmse
  }
}

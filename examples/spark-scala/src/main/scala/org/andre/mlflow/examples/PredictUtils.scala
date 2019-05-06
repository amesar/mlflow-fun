package org.andre.mlflow.examples

import org.apache.spark.sql.{SparkSession,DataFrame}
import org.apache.spark.ml.{PipelineModel,Transformer}
import org.mlflow.api.proto.Service.RunInfo
import org.andre.mlflow.util.MLeapUtils

object PredictionUtils {

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
    val model = MLeapUtils.readModel(modelPath)
    showPredictions(model, data)
  } 

  def showPredictions(model: Transformer, data: DataFrame) {
    val predictions = model.transform(data)
    val df = predictions.select("prediction", "label", "features")
    df.show(10)
  }
}

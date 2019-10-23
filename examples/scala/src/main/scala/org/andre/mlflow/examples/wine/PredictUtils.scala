package org.andre.mlflow.examples.wine

import org.apache.spark.sql.{SparkSession,DataFrame}
import org.apache.spark.ml.{PipelineModel,Transformer}
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLeapUtils

object PredictUtils {

  def predict(client: MlflowClient, runId: String, dataPath: String) {
    val spark = SparkSession.builder.appName("Predict").getOrCreate()
    val data = CommonUtils.readData(spark,dataPath)
    predictSparkML(client, runId, data)
    predictMLeap(client, runId, data)
  }

  def predictSparkML(client: MlflowClient, runId: String, data: DataFrame) {
    println("==== Spark ML")
    val modelPath = client.downloadArtifacts(runId,"spark-model").getAbsolutePath
    val model = PipelineModel.load(modelPath)
    showPredictions(model, data)
  }

  def predictMLeap(client: MlflowClient, runId: String, data: DataFrame) {
    println("==== MLeap")
    val modelPath = "file:" + client.downloadArtifacts(runId,"mleap-model/mleap/model").getAbsolutePath
    val model = MLeapUtils.readModelAsSparkBundle(modelPath)
    showPredictions(model, data)
  } 

  def showPredictions(model: Transformer, data: DataFrame) {
    val predictions = model.transform(data)
    val df = predictions.select(CommonUtils.colFeatures,CommonUtils.colLabel,CommonUtils.colPrediction).sort(CommonUtils.colFeatures,CommonUtils.colLabel,CommonUtils.colPrediction)
    df.show(10)
  }
}

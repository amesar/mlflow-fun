package org.andre.mlflow.examples.wine

import java.io.File
import org.apache.spark.sql.{SparkSession,DataFrame}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.PipelineModel
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLeapUtils

object TrainUtils {
  case class DataHolder(data: DataFrame, trainingData: DataFrame, testData: DataFrame, assembler: VectorAssembler)
  val seed = 2019
  val columnLabel = "quality"

  def readData(spark: SparkSession, dataPath: String) : DataHolder = {
    val data = CommonUtils.readData(spark, dataPath)
    //println("Input Data Schema:")
    //data.printSchema
    val columns = data.columns.toList.filter(_ != columnLabel)

    val assembler = new VectorAssembler()
      .setInputCols(columns.toArray)
      .setOutputCol("features")

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed)
    DataHolder(data, trainingData, testData, assembler)
  }

  def saveModelAsSparkML(client: MlflowClient, runId: String, baseModelDir: String, model: PipelineModel) = {
    val modelPath = s"$baseModelDir/spark-model"
    model.write.overwrite().save(modelPath)
    client.logArtifacts(runId, new File(modelPath), "spark-model")
  }

  def saveModelAsMLeap(client: MlflowClient, runId: String, baseModelDir: String, model: PipelineModel, predictions: DataFrame) = {
    val modelPath = new File(s"$baseModelDir/mleap-model")
    modelPath.mkdir
    MLeapUtils.saveModel(model, predictions, "file:"+modelPath.getAbsolutePath)
    client.logArtifacts(runId, modelPath, "mleap-model/mleap/model") // Make compatible with MLflow Python mlflow.mleap.log_model
  }
}

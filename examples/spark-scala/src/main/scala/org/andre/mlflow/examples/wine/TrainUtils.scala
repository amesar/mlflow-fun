package org.andre.mlflow.examples.wine

import java.io.File
import org.apache.spark.sql.{SparkSession,DataFrame}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.PipelineModel
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLeapUtils

object TrainUtils {
  case class DataHolder(trainingData: DataFrame, testData: DataFrame, assembler: VectorAssembler)
  
  val seed = 2019
  val columnLabel = "quality"

  def readData(spark: SparkSession, dataPath: String) : DataHolder = {
    val data = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(dataPath)
    println("Input Data Schema:")
    data.printSchema
    val columns = data.columns.toList.filter(_ != columnLabel)

    val assembler = new VectorAssembler()
      .setInputCols(columns.toArray)
      .setOutputCol("indexedFeatures")

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed)
    
    DataHolder(trainingData, testData, assembler)
  }

  def saveModelAsSparkML(mlflowClient: MlflowClient, runId: String, baseModelDir: String, model: PipelineModel) = {
    val modelPath = s"$baseModelDir/spark-model"
    model.write.overwrite().save(modelPath)
    mlflowClient.logArtifacts(runId, new File(modelPath), "spark-model")
  }

  def saveModelAsMLeap(mlflowClient: MlflowClient, runId: String, baseModelDir: String, model: PipelineModel, predictions: DataFrame) = {
    val modelPath = new File(s"$baseModelDir/mleap-model")
    modelPath.mkdir
    MLeapUtils.saveModel(model, predictions, "file:"+modelPath.getAbsolutePath)
    mlflowClient.logArtifacts(runId, modelPath, "mleap-model/mleap/model") // Make compatible with MLflow Python mlflow.mleap.log_model
  }
}

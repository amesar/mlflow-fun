package org.andre.mlflow.examples.wine

import java.io.File
import java.nio.file.{Paths,Files}
import org.apache.spark.sql.{SparkSession,DataFrame}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.PipelineModel
import org.mlflow.tracking.{MlflowClient,ActiveRun}
import org.andre.mlflow.util.MLeapUtils

object TrainUtils {
  case class DataHolder(data: DataFrame, trainingData: DataFrame, testData: DataFrame, assembler: VectorAssembler)
  val seed = 2019
  val columnLabel = "quality"

  def readData(spark: SparkSession, dataPath: String) : DataHolder = {
    val data = CommonUtils.readData(spark, dataPath)
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

  def saveModelAsSparkMLContext(run: ActiveRun, baseModelDir: String, model: PipelineModel) = {
    val modelPath = s"$baseModelDir/spark-model"
    model.write.overwrite().save(modelPath)
    run.logArtifacts(Paths.get(modelPath), "spark-model")
  }

  def saveModelAsMLeapContext(run: ActiveRun, baseModelDir: String, model: PipelineModel, predictions: DataFrame) = {
    val modelDir = Paths.get(s"$baseModelDir/mleap-model")
    Files.createDirectories(modelDir)
    MLeapUtils.saveModel(model, predictions, "file:"+modelDir.toAbsolutePath.toString)
    run.logArtifacts(modelDir, "mleap-model/mleap/model") // Make compatible with MLflow Python mlflow.mleap.log_model
  }
}

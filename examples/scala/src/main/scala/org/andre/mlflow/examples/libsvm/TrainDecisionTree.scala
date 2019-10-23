package org.andre.mlflow.examples.libsvm

// From: https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/DecisionTreeRegressionExample.scala

import java.io.{File,PrintWriter}
import org.apache.spark.sql.{SparkSession,DataFrame}
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.feature.{VectorIndexer,VectorIndexerModel}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel,DecisionTreeRegressor}
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus
import com.beust.jcommander.{JCommander, Parameter}
import org.andre.mlflow.util.{MLflowUtils,MLeapUtils}

object TrainDecisionTree {
  case class DataHolder(trainingData: DataFrame, testData: DataFrame, featureIndexer: VectorIndexerModel)
  val spark = SparkSession.builder.appName("DecisionTreeRegressionExample").getOrCreate()
  val seed = 2019

  def main(args: Array[String]) {
    new JCommander(opts, args: _*)
    println("Options:")
    println(s"  Tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  experimentName: ${opts.experimentName}")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  modelPath: ${opts.modelPath}")
    println(s"  maxDepth: ${opts.maxDepth}")
    println(s"  maxBins: ${opts.maxBins}")
    println(s"  runOrigin: ${opts.runOrigin}")

    // MLflow - create or get existing experiment
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)

    val experimentId = MLflowUtils.getOrCreateExperimentId(client, opts.experimentName)
    println("Experiment ID: "+experimentId)

    // Read data
    val dataHolder = readData(opts.dataPath)

    // Train model
    train(client, experimentId, opts.modelPath, opts.maxDepth, opts.maxBins, opts.runOrigin, dataHolder)
  }

  def readData(dataPath: String) : DataHolder = {
    val data = spark.read.format("libsvm").load(dataPath)
    data.printSchema

    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed)
    
    DataHolder(trainingData, testData, featureIndexer)
  }

  def train(client: MlflowClient, experimentId: String, modelPath: String, maxDepth: Int, maxBins: Int, runOrigin: String, dataHolder: DataHolder) {
    // Create a DecisionTree model
    val clf = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
    if (maxDepth != -1) clf.setMaxDepth(maxDepth)
    if (maxBins != -1) clf.setMaxBins(maxBins)

    // MLflow - create run
    val runInfo = client.createRun(experimentId)
    val runId = runInfo.getRunUuid()
    println(s"Run ID: $runId")
    println(s"runOrigin: $runOrigin")

    // MLflow - Log parameters
    client.logParam(runId, "maxDepth",""+clf.getMaxDepth)
    client.logParam(runId, "maxBins",""+clf.getMaxBins)
    client.logParam(runId, "runOrigin",runOrigin)
    println(s"Params:")
    println(s"  maxDepth: ${clf.getMaxDepth}")
    println(s"  maxBins: ${clf.getMaxBins}")

    // MLflow - set tags
    client.setTag(runId, "mlflow.source.name",MLflowUtils.getSourceName(getClass())) // populates "Source" field in UI
    client.setTag(runId, "mlflow.runName","myRun") // populates "Run Name" field in UI

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline().setStages(Array(dataHolder.featureIndexer, clf))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(dataHolder.trainingData)

    // Make predictions.
    val predictions = model.transform(dataHolder.testData)

    // Create metrics: select (prediction, true label) and compute test error.
    val rmse = PredictUtils.evaluatePredictions(predictions)

    // MLflow - Log metric
    client.logMetric(runId, "rmse",rmse)

    // Select example rows to display.
    println("Prediction:")
    predictions.select("prediction", "label", "features").show(5)

    // MLflow - Log tree model artifact
    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    val path="treeModel.txt"
    new PrintWriter(path) { write(treeModel.toDebugString) ; close }
    client.logArtifact(runId,new File(path),"details")

    // MLflow - Save model in Spark ML and MLeap formats
    saveModelAsSparkML(client, runId, modelPath, model)
    saveModelAsMLeap(client, runId, modelPath, model, predictions)

    // MLflow - close run
    client.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
  }

  def saveModelAsSparkML(client: MlflowClient, runId: String, baseModelDir: String, model: PipelineModel) = {
    val modelPath = s"$baseModelDir/spark-model"
    model.write.overwrite().save(modelPath)
    client.logArtifacts(runId, new File(modelPath), "spark-model")
  }

  def saveModelAsMLeap(client: MlflowClient, runId: String, baseModelDir: String, model: PipelineModel, predictions: DataFrame) = {
    val modelPath = new File(s"$baseModelDir/mleap-model")
    modelPath.mkdir
    MLeapUtils.saveModelAsSparkBundle("file:"+modelPath.getAbsolutePath, model, predictions)
    client.logArtifacts(runId, modelPath, "mleap-model/mleap/model") // Make compatible with MLflow Python mlflow.mleap.log_model
  }

  object opts {
    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null

    @Parameter(names = Array("--modelPath" ), description = "Data path", required=true)
    var modelPath: String = null

    @Parameter(names = Array("--maxDepth" ), description = "maxDepth", required=false)
    var maxDepth = -1

    @Parameter(names = Array("--maxBins" ), description = "maxBins", required=false)
    var maxBins = -1

    @Parameter(names = Array("--runOrigin" ), description = "runOrigin", required=false)
    var runOrigin = "None"

    @Parameter(names = Array("--experimentName" ), description = "experimentName", required=false)
    var experimentName = "scala_DecisionTree"
  }
}

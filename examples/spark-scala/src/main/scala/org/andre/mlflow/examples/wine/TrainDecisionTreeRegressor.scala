package org.andre.mlflow.examples.wine

// From: https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/DecisionTreeRegressionExample.scala

import java.io.{File,PrintWriter}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.{DecisionTreeRegressor,DecisionTreeRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus
import com.beust.jcommander.{JCommander, Parameter}
import org.andre.mlflow.util.MLflowUtils

object TrainDecisionTreeRegressor {
  val spark = SparkSession.builder.appName("DecisionTreeRegressionExample").getOrCreate()
  val columnLabel = "quality"

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
    val mlflowClient = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)

    val experimentId = MLflowUtils.setExperiment(mlflowClient, opts.experimentName)
    println("Experiment ID: "+experimentId)

    // Read data
    val dataHolder = TrainUtils.readData(spark, opts.dataPath)
    println("dataHolder: "+dataHolder)

    // Train model
    train(mlflowClient, experimentId, opts.modelPath, opts.maxDepth, opts.maxBins, opts.runOrigin, dataHolder)
  }

  def train(mlflowClient: MlflowClient, experimentId: String, modelPath: String, maxDepth: Int, maxBins: Int, runOrigin: String, dataHolder: TrainUtils.DataHolder) {
    // MLflow - create run
    val sourceName = (getClass().getSimpleName()+".scala").replace("$","")
    val runInfo = mlflowClient.createRun(experimentId, sourceName);
    val runId = runInfo.getRunUuid()
    println(s"Run ID: $runId")
    println(s"runOrigin: $runOrigin")

    // Create model
    val clf = new DecisionTreeRegressor()
      .setLabelCol(columnLabel)
      .setFeaturesCol("indexedFeatures")
    if (maxDepth != -1) clf.setMaxDepth(maxDepth)
    if (maxBins != -1) clf.setMaxBins(maxBins)

    // MLflow - Log parameters
    mlflowClient.logParam(runId, "maxDepth",""+clf.getMaxDepth)
    mlflowClient.logParam(runId, "maxBins",""+clf.getMaxBins)
    mlflowClient.logParam(runId, "runOrigin",runOrigin)
    println(s"Params:")
    println(s"  maxDepth: ${clf.getMaxDepth}")
    println(s"  maxBins: ${clf.getMaxBins}")


    // Chain indexer and tree in a Pipeline
    val pipeline = new Pipeline().setStages(Array(dataHolder.assembler,clf))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(dataHolder.trainingData)

    // Make predictions.
    val predictions = model.transform(dataHolder.testData)
    println("Predictions Schema:")
    predictions.printSchema

    // Create metrics: select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol(columnLabel)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Metrics:")
    println(s"  RMSE: $rmse")
    println(s"  isLargerBetter: ${evaluator.isLargerBetter}")

    // MLflow - Log metric
    mlflowClient.logMetric(runId, "rmse",rmse)

    // Select example rows to display.
    println("Prediction:")
    predictions.select("prediction", columnLabel, "indexedFeatures").show(5)


    // MLflow - Log tree model artifact
    val treeModel = model.stages.last.asInstanceOf[DecisionTreeRegressionModel]
    val path="treeModel.txt"
    new PrintWriter(path) { write(treeModel.toDebugString) ; close }
    mlflowClient.logArtifact(runId,new File(path),"details")

    // MLflow - Save model in Spark ML and MLeap formats
    TrainUtils.saveModelAsSparkML(mlflowClient, runId, modelPath, model)
    TrainUtils.saveModelAsMLeap(mlflowClient, runId, modelPath, model, predictions)

    // MLflow - close run
    mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
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

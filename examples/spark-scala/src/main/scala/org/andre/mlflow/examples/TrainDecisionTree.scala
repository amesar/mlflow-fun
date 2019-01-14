package org.andre.mlflow.examples

// From: https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/DecisionTreeRegressionExample.scala

import java.io.{File,PrintWriter}
import scala.collection.JavaConversions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel,DecisionTreeRegressor}
import org.apache.spark.sql.SparkSession
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus
import com.beust.jcommander.{JCommander, Parameter}

object TrainDecisionTree {
  val seed = 2019

  def main(args: Array[String]) {
    println("args: "+args.toList.mkString(" "))
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  Tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  experimentName: ${opts.experimentName}")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  modelPath: ${opts.modelPath}")
    println(s"  maxDepth: ${opts.maxDepth}")
    println(s"  maxBins: ${opts.maxBins}")
    println(s"  runOrigin: ${opts.runOrigin}")
    val mlflowClient = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    val spark = SparkSession.builder.appName("DecisionTreeRegressionExample").getOrCreate()
    train(spark, mlflowClient, opts.dataPath, opts.modelPath, opts.maxDepth, opts.maxBins, opts.runOrigin)
  }

  def train(spark: SparkSession, dataPath: String, modelPath: String, maxDepth: Int, maxBins: Int, runOrigin: String = "") {
    val mlflowClient = new MlflowClient()
    train(spark, mlflowClient, dataPath, modelPath, maxDepth, maxBins, runOrigin)
  }

  def train(spark: SparkSession, mlflowClient: MlflowClient, dataPath: String, modelPath: String, maxDepth: Int, maxBins: Int, runOrigin: String) {
    val data = spark.read.format("libsvm").load(dataPath)

    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed)

    // MLflow - create or get existing experiment
    val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, opts.experimentName)
    //println("Experiment name: "+expName)
    println("Experiment ID: "+expId)

    // Train a DecisionTree model.
    val clf = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    if (maxDepth != -1) clf.setMaxDepth(maxDepth)
    if (maxBins != -1) clf.setMaxBins(maxBins)
    println(s"clf.MaxDepth: ${clf.getMaxDepth}")
    println(s"clf.MaxBins: ${clf.getMaxBins}")

    // MLflow - create run
    val sourceName = (getClass().getSimpleName()+".scala").replace("$","")
    val runInfo = mlflowClient.createRun(expId, sourceName);
    val runId = runInfo.getRunUuid()
    println(s"Run ID: $runId")
    println(s"runOrigin: $runOrigin")

    // MLflow - Log parameters
    mlflowClient.logParam(runId, "maxDepth",""+clf.getMaxDepth)
    mlflowClient.logParam(runId, "maxBins",""+clf.getMaxBins)
    mlflowClient.logParam(runId, "runOrigin",runOrigin)

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline().setStages(Array(featureIndexer, clf))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    println("Prediction:")
    predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)

    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    println(s"isLargerBetter: ${evaluator.isLargerBetter}")

    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println(s"Learned regression tree model:\n ${treeModel.toDebugString}")

    // MLflow - Log metric
    mlflowClient.logMetric(runId, "rmse",rmse.toFloat)

    // MLflow - Log simple artifact
    val path="info.txt"
    new PrintWriter(path) { write("Info: "+new java.util.Date()) ; close }
    mlflowClient.logArtifact(runId,new File(path),"info")

    // MLflow - save model as Spark ML artifact
    val sparkModelPath = s"$modelPath/spark_model"
    //model.save(sparkModelPath)
    model.write.overwrite().save(sparkModelPath)
    mlflowClient.logArtifacts(runId, new File(sparkModelPath), "spark_model")

    // MLflow - save model as MLeap artifact
    val mleapModelDir = new File(s"$modelPath/mleap_model")
    mleapModelDir.mkdir
    MLeapUtils.saveModel(model, predictions, "file:"+mleapModelDir.getAbsolutePath)
    mlflowClient.logArtifacts(runId, mleapModelDir, "mleap_model")

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
    var experimentName = "scala/SimpleDecisionTree"
  }
}

package org.andre.mlflow.examples.wine

import java.nio.file.{Paths,Files}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.{DecisionTreeRegressor,DecisionTreeRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.mlflow.tracking.{MlflowClient,MlflowContext}
import com.beust.jcommander.{JCommander, Parameter}
import org.andre.mlflow.util.MLflowUtils

/**
 * MLflow DecisionTreeRegressor with wine quality data.
 */
object TrainDecisionTreeRegressorContext {
  val spark = SparkSession.builder.appName("DecisionTreeRegressionExample").getOrCreate()
  val metrics = Seq("rmse","r2", "mae")

  def main(args: Array[String]) {
    new JCommander(opts, args: _*)
    println("Options:")
    println(s"  trackingUri: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  experimentName: ${opts.experimentName}")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  outputDir: ${opts.outputDir}")
    println(s"  maxDepth: ${opts.maxDepth}")
    println(s"  maxBins: ${opts.maxBins}")
    println(s"  runOrigin: ${opts.runOrigin}")

    // MLflow - create or get existing experiment
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    val experimentId = MLflowUtils.getOrCreateExperimentId(client, opts.experimentName)
    println("Experiment ID: "+experimentId)

    // Read data
    val dataHolder = TrainUtils.readData(spark, opts.dataPath)
    println("dataHolder: "+dataHolder)

    // Train model
    train(client, opts.experimentName, opts.outputDir, opts.maxDepth, opts.maxBins, opts.runOrigin, dataHolder)
  }

  def train(client: MlflowClient, experimentName: String, outputDir: String, maxDepth: Int, maxBins: Int, runOrigin: String, dataHolder: TrainUtils.DataHolder) {
    // MLflow context - set current experiment
    val mlflow = new MlflowContext(client)
    mlflow.setExperimentName(experimentName)

    // MLflow - create run
    val run = mlflow.startRun("Wine Context")
    val runId = run.getId()
    println(s"Run ID: $runId")
    println(s"runOrigin: $runOrigin")

    // MLflow - set tag
    run.setTag("dataPath",dataHolder.dataPath)

    // MLflow - log parameters
    val params = Seq(("maxDepth",maxDepth),("maxBins",maxBins),("runOrigin",runOrigin))
    println(s"Params:")
    for (p <- params) {
      println(s"  ${p._1}: ${p._2}")
      run.logParam(p._1,p._2.toString)
    }

    // Create model
    val dt = new DecisionTreeRegressor()
      .setLabelCol(CommonUtils.colLabel)
      .setFeaturesCol(CommonUtils.colFeatures)
      .setMaxDepth(maxDepth)
      .setMaxBins(maxBins)

    // Create pipeline
    val pipeline = new Pipeline().setStages(Array(dataHolder.assembler,dt))

    // Fit model
    val model = pipeline.fit(dataHolder.trainingData)

    // Make predictions
    val predictions = model.transform(dataHolder.testData)

    // MLflow - log metrics
    println("Metrics:")
    for (metric <- metrics) {
      val evaluator = new RegressionEvaluator()
        .setLabelCol(CommonUtils.colLabel)
        .setPredictionCol(CommonUtils.colPrediction)
        .setMetricName(metric)
      val v = evaluator.evaluate(predictions)
      println(s"  $metric: $v - isLargerBetter: ${evaluator.isLargerBetter}")
      run.logMetric(metric, v)
    }

    // MLflow - log tree model artifact
    val odir = Paths.get(outputDir)
    Files.createDirectories(odir)
    val treeModel = model.stages.last.asInstanceOf[DecisionTreeRegressionModel]
    val path= Paths.get(odir.toString,"treeModel.txt")
    Files.write(path,treeModel.toDebugString.getBytes)
    run.logArtifact(path,"details")

    // MLflow - Save model in Spark ML and MLeap formats
    TrainUtils.logModelAsSparkMLContext(run, s"$outputDir/models", model)
    TrainUtils.logModelAsMLeapContext(run, s"$outputDir/models", model, predictions)

    // MLflow - close run
    run.endRun()
  }

  object opts {
    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null

    @Parameter(names = Array("--outputDir" ), description = "Output directory", required=false)
    var outputDir: String = "tmp"

    @Parameter(names = Array("--maxDepth" ), description = "maxDepth param", required=false)
    var maxDepth: Int = 5 // per doc

    @Parameter(names = Array("--maxBins" ), description = "maxBins param", required=false)
    var maxBins: Int = 32 // per doc

    @Parameter(names = Array("--runOrigin" ), description = "runOrigin tag", required=false)
    var runOrigin = "None"

    @Parameter(names = Array("--experimentName" ), description = "Experiment name", required=false)
    var experimentName = "scala_context"
  }
}

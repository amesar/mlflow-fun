package org.andre.mlflow.examples.wine

import java.io.{File,PrintWriter}
import com.beust.jcommander.{JCommander, Parameter}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLflowUtils

/**
 * Trains and predicts MLflow DecisionTreeRegressor with wine quality data.
 */
object TrainAndPredict {
  val spark = SparkSession.builder.appName("DecisionTreeRegressionExample").getOrCreate()
  MLflowUtils.showVersions(spark)
  val metrics = Seq("rmse","r2", "mae")

  def main(args: Array[String]) {
    new JCommander(opts, args: _*)
    println("Options:")
    println(s"  trackingUri: ${opts.trackingUri}")
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

    println("****** Train")
    // Read data
    val dataHolder = TrainUtils.readData(spark, opts.dataPath)
    println("dataHolder: "+dataHolder)

    // Train model
    TrainDecisionTreeRegressor.train(client, experimentId, opts.modelPath, opts.maxDepth, opts.maxBins, opts.runOrigin, dataHolder)

    // Predict model
    println("****** Predict")
    val runInfo = MLflowUtils.getLastRunInfo(client, experimentId)
    PredictUtils.predict(client, runInfo.getRunId, opts.dataPath)
  }

  object opts {
    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null

    @Parameter(names = Array("--modelPath" ), description = "Model path", required=true)
    var modelPath: String = null

    @Parameter(names = Array("--maxDepth" ), description = "maxDepth param", required=false)
    var maxDepth: Int = 5 // per doc

    @Parameter(names = Array("--maxBins" ), description = "maxBins param", required=false)
    var maxBins: Int = 32 // per doc

    @Parameter(names = Array("--runOrigin" ), description = "runOrigin tag", required=false)
    var runOrigin = "None"

    @Parameter(names = Array("--experimentName" ), description = "Experiment name", required=false)
    var experimentName = "scala_classic"
  }
}

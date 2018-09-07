package org.andre.mlflow.examples

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.SparkSession
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds.BasicMlflowHostCreds
import org.mlflow.api.proto.Service.RunStatus
import scala.collection.JavaConversions._

object DecisionTreeRegressionExample {
  val expName = "scala/SimpleDecisionTreeRegression"

  def main(args: Array[String]): Unit = {
    if (args.length < 4) {
      println("ERROR: Expecting TRACKING_SERVER_URI DATA_PATH MAX_DEPTH MAX_BINS (TOKEN)")
      System.exit(1)
    }

    val trackingUri = args(0)
    println(s"Tracking URI: $trackingUri")
    val mlflowClient = 
      if (args.length > 4) {
        new MlflowClient(new BasicMlflowHostCreds(trackingUri,args(4)))
      } else {
        new MlflowClient(trackingUri)
      }
    val spark = SparkSession.builder.appName("DecisionTreeRegressionExample").getOrCreate()
    train(mlflowClient, spark,  args(1), args(2).toInt, args(3).toInt)
  }

  def train(mlflowClient: MlflowClient, spark: SparkSession, dataPath: String, maxDepth: Int, maxBins: Int) {
    val data = spark.read.format("libsvm").load(dataPath)

    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // MLflow - create or get existing experiment
    val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)
    println("Experiment name: "+expName)
    println("Experiment ID: "+expId)

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    if (maxDepth != -1) dt.setMaxDepth(maxDepth)
    if (maxBins != -1) dt.setMaxBins(maxBins)
    println(s"dt.MaxDepth: ${dt.getMaxDepth}")
    println(s"dt.MaxBins: ${dt.getMaxBins}")

    // MLflow - create run
    val sourceName = getClass().getSimpleName()+".scala"
    val runInfo = mlflowClient.createRun(expId, sourceName);
    val runId = runInfo.getRunUuid()

    // MLflow - Log parameters
    mlflowClient.logParam(runId, "maxDepth",""+dt.getMaxDepth)
    mlflowClient.logParam(runId, "maxBins",""+dt.getMaxBins)

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))

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

    // MLflow - close run
    mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
  }
}

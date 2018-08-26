package org.andre.mlflow.examples

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.SparkSession
import com.databricks.mlflow.client2.ApiClient;
import com.databricks.mlflow.client2.RunContext;
import com.databricks.mlflow.client2.objects._

object DecisionTreeRegressionExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("DecisionTreeRegressionExample").getOrCreate()

    val apiClient = new ApiClient(args(0))

    // Load the data stored in LIBSVM format as a DataFrame
    val dataPath = args(1)

    val maxDepth = if (args.length > 2) args(2).toInt else -1
    val maxBins = if (args.length > 3) args(3).toInt else -1

    train(apiClient, spark, dataPath,maxDepth,maxBins)
  }

  def train(apiClient: ApiClient, spark: SparkSession, dataPath: String, maxDepth: Int, maxBins: Int) {

    // MLflow - create or get existing experiment
    val expName = "scala/DecisionTreeRegressionExample"
    val expId = apiClient.getOrCreateExperimentId(expName)
    println("Experiment name: "+expName)
    println("Experiment ID: "+expId)

    // MLflow - create new run
    val runContext = new RunContext(apiClient, expId, "run", "LOCAL", "DecisionTreeRegressionExample.scala", "doe")

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

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    if (maxDepth != -1) dt.setMaxDepth(maxDepth)
    if (maxBins != -1) dt.setMaxBins(maxBins)
    println(s"dt.MaxDepth: ${dt.getMaxDepth}")
    println(s"dt.MaxBins: ${dt.getMaxBins}")
    
    // MLflow - Log parameters
    runContext.logParameter("maxDepth",""+dt.getMaxDepth)
    runContext.logParameter("maxBins",""+dt.getMaxBins)

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dt))

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

    // MLflow - Log metric
    runContext.logMetric("rmse",rmse)

    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    println(s"isLargerBetter: ${evaluator.isLargerBetter}")

    // MLflow - Log artifact file
    val localFile = "explainParams.txt"
    new java.io.PrintWriter(localFile) { write(evaluator.explainParams); close }
    runContext.logArtifact(localFile,"")

    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println(s"Learned regression tree model:\n ${treeModel.toDebugString}")
  
    runContext.close()
  }
}

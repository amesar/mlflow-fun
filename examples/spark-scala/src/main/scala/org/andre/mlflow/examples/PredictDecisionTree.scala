package org.andre.mlflow.examples

import com.beust.jcommander.{JCommander, Parameter}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.PipelineModel
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds.BasicMlflowHostCreds

object PredictDecisionTree {

  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  runId: ${opts.runId}")

    val mlflowClient =
      if (opts.token != null) {
        new MlflowClient(new BasicMlflowHostCreds(opts.trackingUri,opts.token))
      } else {
        new MlflowClient(opts.trackingUri)
      }
    val spark = SparkSession.builder.appName("Predict").getOrCreate()
    val data = spark.read.format("libsvm").load(opts.dataPath)

    val runInfo = mlflowClient.getRun(opts.runId).getInfo
    val uri = runInfo.getArtifactUri

    println("==== Spark ML")
    val modelPath = s"${uri}/spark_model"
    val model = PipelineModel.load(modelPath)
    val predictions = model.transform(data)
    val df = predictions.select("prediction", "label", "features")
    df.show(10)

    println("==== Mleap")
    val modelPath2 = s"file:${uri}/mleap_model"
    val model2 = MLeapUtils.readModel(modelPath2)
    val predictions2 = model2.transform(data)
    val df2 = predictions2.select("prediction", "label", "features")
    df2.show(10)
  }

  object opts {
    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null

    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=true)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--runId" ), description = "runId", required=true)
    var runId: String = null
  }
}

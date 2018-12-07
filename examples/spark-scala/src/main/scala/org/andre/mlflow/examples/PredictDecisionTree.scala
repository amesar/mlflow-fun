package org.andre.mlflow.examples

import com.beust.jcommander.{JCommander, Parameter}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.PipelineModel

object PredictDecisionTree {
  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  modelPath: ${opts.modelPath}")

    val spark = SparkSession.builder.appName("Predict").getOrCreate()
    val data = spark.read.format("libsvm").load(opts.dataPath)

    val model = PipelineModel.load(opts.modelPath)
    val predictions = model.transform(data)
    println("Prediction:")
    predictions.select("prediction", "label", "features").show(10,false)
  }

  object opts {
    @Parameter(names = Array("--modelPath" ), description = "Data path", required=true)
    var modelPath: String = null

    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null
  }
}

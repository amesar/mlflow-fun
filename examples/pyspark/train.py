"""
PySpark Decision Tree Classification Example.
"""
from __future__ import print_function

import sys,os
from argparse import ArgumentParser
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import mlflow
from mlflow import version
from mlflow import spark as mlflow_spark

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

splits = [0.7, 0.3]
default_seed = 2019
metric_names = ["accuracy","f1","weightedPrecision"]

def train(max_depth, max_bins, random):
    print("Parameters: max_depth: {}  max_bins: {}  random: {}".format(max_depth,max_bins,random))
    spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

    # Load the data stored in LIBSVM format as a DataFrame.
    data_path = "../data/sample_libsvm_data.txt"
    data = spark.read.format("libsvm").load(data_path)

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets
    if random:
        (trainingData, testData) = data.randomSplit(splits)
    else:
        (trainingData, testData) = data.randomSplit(splits, default_seed)

    # Train a DecisionTree model.
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("max_bins",max_bins)
    mlflow.log_param("random",random)
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=max_depth, maxBins=max_bins)

    # Chain indexers and tree in a Pipeline.
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Get evaluator metrics
    print("Metrics:")
    for metric_name in metric_names:
        evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName=metric_name)
        metric_value = evaluator.evaluate(predictions)
        print("  {}: {}".format(metric_name,metric_value))
        mlflow.log_metric(metric_name,metric_value)

    treeModel = model.stages[2]
    print(treeModel)

    mlflow_spark.log_model(model, "spark-model")
    #mlflow.mleap.log_model(model, testData, "mleap-model") # TODO: Bombs :(


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", default="pyspark", required=False)
    parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=2, type=int)
    parser.add_argument("--max_bins", dest="max_bins", help="max_bins", default=32, type=int)
    parser.add_argument("--random", dest="random", help="Random", required=False, default=False, action='store_true')

    args = parser.parse_args()
    print("MLflow Version:", version.VERSION)

    client = mlflow.tracking.MlflowClient()
    print("experiment_name:",args.experiment_name)
    mlflow.set_experiment(args.experiment_name)
    print("experiment_id:",client.get_experiment_by_name(args.experiment_name).experiment_id)

    with mlflow.start_run() as run:
        print("run_id:",run.info.run_uuid)
        print("experiment_id:",run.info.experiment_id)
        train(args.max_depth,args.max_bins,args.random)

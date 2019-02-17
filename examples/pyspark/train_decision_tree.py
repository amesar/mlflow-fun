"""
PySpark Decision Tree Classification Example.
"""
from __future__ import print_function

import sys,os
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
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME","py/spark/DecisionTree")
print("experiment_name:",experiment_name)
mlflow.set_experiment(experiment_name)

def run(max_depth,max_bins):
    print("max_depth={} max_bins={}".format(max_depth,max_bins))
    spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

    # Load the data stored in LIBSVM format as a DataFrame.
    dir = str(os.environ['SPARK_HOME'])
    path = os.path.join(dir,"data/mllib/sample_libsvm_data.txt")
    data = spark.read.format("libsvm").load(path)

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("max_bins",max_bins)
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=max_depth, maxBins=max_bins)

    # Chain indexers and tree in a Pipeline.
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error.
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    test_error = 1.0 - accuracy
    print("Test Error = {} ".format(test_error))

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("test_error", test_error)

    treeModel = model.stages[2]
    print(treeModel)

    mlflow_spark.log_model(model, "spark-model")

    spark.stop()

if __name__ == "__main__":
    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    max_bins = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    current_file = os.path.basename(__file__)
    print("MLflow Version:", version.VERSION)

    client = mlflow.tracking.MlflowClient()
    print("experiment_id:",client.get_experiment_by_name(experiment_name).experiment_id)

    with mlflow.start_run(source_name=current_file):
        run(max_depth,max_bins)

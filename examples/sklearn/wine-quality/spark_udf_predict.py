"""
Serve predictions with Spark UDF.
"""
from __future__ import print_function

import sys
from pyspark.sql import SparkSession
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    run_id = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "../../data/wine-quality/wine-quality-red.csv"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "sklearn-model"
    print("path:",path)
    print("run_id:",run_id)
    print("model_name:",model_name)
    print("MLflow Version:", mlflow.version.VERSION)

    spark = SparkSession.builder.appName("ServePredictions").getOrCreate()

    df = spark.read.option("inferSchema",True).option("header", True).csv(path) if path.endswith(".csv") \
    else spark.read.option("multiLine",True).json(path)

    if "quality" in df.columns:
        df = df.drop("quality")
    df.show(10)

    udf = mlflow.pyfunc.spark_udf(spark, f"runs:/{run_id}/sklearn-model")
    predictions = df.withColumn("prediction", udf(*df.columns))
    predictions.show(10)
    predictions.select("prediction").show(10)

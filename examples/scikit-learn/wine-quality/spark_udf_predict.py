"""
Serve predictions with Spark UDF.
"""
from __future__ import print_function

import sys
from pyspark.sql import SparkSession
import mlflow
import mlflow.sklearn
experiment_name = "py/sk/ElasticNet/WineQuality"

if __name__ == "__main__":
    path = "wine-quality.csv"
    run_id = sys.argv[1]
    print("MLflow Version:", mlflow.version.VERSION)

    spark = SparkSession.builder.appName("ServePredictions").getOrCreate()
    df = spark.read.option("inferSchema",True).option("header", True).csv(path)
    df = df.drop("quality")
    df.show(10)

    udf = mlflow.pyfunc.spark_udf(spark, "model", run_id=run_id)
    df2 = df.withColumn("prediction", udf(*df.columns))
    df2.show(10)

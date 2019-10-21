import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from common import *

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="run_id", required=True)
    parser.add_argument("--data_path", dest="data_path", help="data_path", required=True)
    parser.add_argument("--udf_predict", dest="udf_predict", help="Predict with UDF", default=False, action='store_true')
    parser.add_argument("--udf_workaround", dest="udf_workaround", help="UDF workaround", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print("  {}: {}".format(arg,getattr(args, arg)))

    spark = SparkSession.builder.appName("Predict").getOrCreate()
    data = read_data(spark, args.data_path)

    # Predict with Spark ML
    model_uri = f"runs:/{args.run_id}/spark-model"
    print("model_uri:",model_uri)
    model = mlflow.spark.load_model(model_uri)
    predictions = model.transform(data)
    df = predictions.select(colPrediction, colLabel, colFeatures)
    print("Spark ML predictions")
    df.show(5,False)

    # Predict with UDF
    if args.udf_predict:
        if args.udf_workaround:
            model_uri = f"runs:/{args.run_id}/udf-spark-model"
            print("UDF model_uri:",model_uri)
            print("UDF predictions - with workaround")
            udf = mlflow.pyfunc.spark_udf(spark, model_uri)
            predictions = data.withColumn("prediction", udf(*data.columns))
            predictions.show(5,False)
            df = predictions.select(colPrediction, colLabel)
        else:
            print("UDF predictions - with no workaround")
            udf = mlflow.pyfunc.spark_udf(spark, model_uri)
            predictions = data.withColumn("prediction", udf(*data.columns))
            predictions.show(5,False)

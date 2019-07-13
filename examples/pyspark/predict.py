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
    args = parser.parse_args()
    print("Arguments:")
    print("  run_id:",args.run_id)
    print("  data_path:",args.data_path)

    spark = SparkSession.builder.appName("Predict").getOrCreate()
    data = read_data(spark, args.data_path)

    model_uri = "runs:/{}/spark-model".format(args.run_id)
    print("model_uri:",model_uri)

    model = mlflow.spark.load_model(model_uri)
    predictions = model.transform(data)
    df = predictions.select(colPrediction, colLabel, colFeatures)
    print("Predictions:")
    df.show(5,False)

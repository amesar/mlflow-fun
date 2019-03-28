
import os,sys
import mlflow
import mlflow.spark as mlflow_spark
from pyspark.sql import SparkSession

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Predict").getOrCreate()
    run_id = sys.argv[1]

    dir = str(os.environ['SPARK_HOME'])
    path = os.path.join(dir,"data/mllib/sample_libsvm_data.txt")
    data = spark.read.format("libsvm").load(path)

    model = mlflow_spark.load_model("spark-model", run_id=run_id)
    predictions = model.transform(data)

    print("Prediction Dataframe")
    predictions.printSchema()

    print("Filtered Prediction Dataframe")
    df = predictions.select("prediction", "indexedLabel","probability").filter("prediction <> indexedLabel")
    df.printSchema()
    df.show(5,False)

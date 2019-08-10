
def read_data(spark, data_path):
    data_path = data_path.replace("/dbfs","dbfs:")
    return spark.read.csv(data_path, header="true", inferSchema="true")

colLabel = "quality"
colPrediction = "prediction"
colFeatures = "features"


def read_data(spark, data_path):
    return spark.read.format("csv") \
      .option("header", "true") \
      .option("inferSchema", "true") \
      .load(data_path.replace("/dbfs","dbfs:"))

colLabel = "quality"
colPrediction = "prediction"
colFeatures = "features"

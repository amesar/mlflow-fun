# Databricks notebook source
# MAGIC %md # Simple PySpark MLflow train and predict notebook
# MAGIC * Predicts as Spark ML and UDF 

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("maxDepth", "2")
dbutils.widgets.text("maxBins", "32")
maxDepth = int(dbutils.widgets.get("maxDepth"))
maxBins = float(dbutils.widgets.get("maxBins"))
maxDepth, maxBins

# COMMAND ----------

import mlflow
import mlflow.spark
print("MLflow Version:", mlflow.version.VERSION)

# COMMAND ----------

metrics = ["rmse","r2", "mae"]
colLabel = "quality"
colPrediction = "prediction"
colFeatures = "features"

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data_path = download_wine_file()

# COMMAND ----------

data = spark.read.format("csv") \
  .option("header", "true") \
  .option("inferSchema", "true") \
  .load(data_path.replace("/dbfs","dbfs:")) 
(trainingData, testData) = data.randomSplit([0.7, 0.3], 2019)

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

with mlflow.start_run() as run:
    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id
    print("run_id:",run_id)
    print("experiment_id:",experiment_id)  
    
    # Log MLflow parameters
    print("Parameters:")
    print("  maxDepth:",maxDepth)
    print("  maxBins:",maxBins)
    mlflow.log_param("maxDepth",maxDepth)
    mlflow.log_param("maxBins",maxBins)
    
    # Create model
    dt = DecisionTreeRegressor(labelCol=colLabel, featuresCol=colFeatures, \
                               maxDepth=maxDepth, maxBins=maxBins)

    # Create pipeline
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=colFeatures)
    pipeline = Pipeline(stages=[assembler, dt])
    
    # Fit model
    model = pipeline.fit(trainingData)
    
    # Log MLflow training metrics
    print("Metrics:")
    predictions = model.transform(testData)
    for metric in metrics:
        evaluator = RegressionEvaluator(labelCol=colLabel, predictionCol=colPrediction, metricName=metric)
        v = evaluator.evaluate(predictions)
        print("  {}: {}".format(metric,v))
        mlflow.log_metric(metric,v)
    
    # Log MLflow model
    mlflow.spark.log_model(model, "spark-model")

# COMMAND ----------

display_run_uri(experiment_id, run_id)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = "runs:/{}/spark-model".format(run_id)
model_uri

# COMMAND ----------

# MAGIC %md #### Predict as Spark ML

# COMMAND ----------

model = mlflow.spark.load_model(model_uri)
predictions = model.transform(data)
display(predictions.select(colPrediction, colLabel, colFeatures))

# COMMAND ----------

# MAGIC %md #### Predict as UDF

# COMMAND ----------

udf = mlflow.pyfunc.spark_udf(spark, model_uri)
data.withColumn("prediction", udf(*data.columns))
display(predictions)
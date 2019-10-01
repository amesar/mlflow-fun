# Databricks notebook source
# MAGIC %md ## MLflow Advanced PySpark Train
# MAGIC 
# MAGIC * Libraries:
# MAGIC   * PyPI package: mlflow 
# MAGIC   * Maven coordinates: ml.combust.mleap:mleap-spark_2.11:0.13.0
# MAGIC * Synopsis:
# MAGIC   * Model: DecisionTreeRegressor
# MAGIC   * Data: Wine quality
# MAGIC   * Logs  model in Spark ML and MLeap format
# MAGIC * Prediction: [Predict_PySpark_Model](https://demo.cloud.databricks.com/#notebook/3538261)
# MAGIC * Experiment: 2972392 - [/Users/andre.mesarovic@databricks.com/gd_pyspark](https://demo.cloud.databricks.com/#mlflow/experiments/2972392)

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./UtilsTrain

# COMMAND ----------

dbutils.widgets.text(WIDGET_TRAIN_EXPERIMENT, create_experiment_name("pyspark"))
dbutils.widgets.text("maxDepth", "2")
dbutils.widgets.text("maxBins", "32")

experiment_name = dbutils.widgets.get(WIDGET_TRAIN_EXPERIMENT)
maxDepth = int(dbutils.widgets.get("maxDepth"))
maxBins = int(dbutils.widgets.get("maxBins"))

maxDepth, maxBins, experiment_name

# COMMAND ----------

if not experiment_name.isspace():
    set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data = read_wine_data()
(trainingData, testData) = data.randomSplit([0.7, 0.3], 2019)

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import mlflow.spark

metrics = ["rmse","r2", "mae"]

with mlflow.start_run() as run:
    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",experiment_id)
    
    # MLflow - log parameters
    print("Parameters:")
    print("  maxDepth:",maxDepth)
    print("  maxBins:",maxBins)
    mlflow.log_param("maxDepth",maxDepth)
    mlflow.log_param("maxBins",maxBins)

    # Create pipeline
    dt = DecisionTreeRegressor(labelCol=colLabel, featuresCol=colFeatures, maxDepth=maxDepth, maxBins=maxBins)
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=colFeatures)
    pipeline = Pipeline(stages=[assembler, dt])
    
    # Fit model and predict
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)

    # MLflow - log metrics
    print("Metrics:")
    predictions = model.transform(testData)
    for metric in metrics:
        evaluator = RegressionEvaluator(labelCol=colLabel, predictionCol=colPrediction, metricName=metric)
        v = evaluator.evaluate(predictions)
        print("  {}: {}".format(metric,v))
        mlflow.log_metric(metric,v)

    # MLflow - log models
    mlflow.spark.log_model(model, "spark-model")
    mlflow.mleap.log_model(spark_model=model, sample_input=testData, artifact_path="mleap-model")

# COMMAND ----------

# MAGIC %md ### Result

# COMMAND ----------

display_run_uri(run.info.experiment_id, run.info.run_id)

# COMMAND ----------

dbutils.notebook.exit(run.info.run_id+" "+run.info.experiment_id)
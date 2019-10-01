# Databricks notebook source
# MAGIC %md ## Train PySpark Model with MLflow using trackMLlib
# MAGIC 
# MAGIC * Cross Validation `trackMLlib` notes:
# MAGIC   * See: [MLlib+MLflow Integration](https://docs.databricks.com/spark/latest/mllib/mllib-mlflow-integration.html) documentation page.
# MAGIC   * If the "Show parent metrics" widget is true, then parent run metrics will be calculated and displayed with leading `_` as in `_accuracy` 
# MAGIC * Libraries:
# MAGIC   * PyPI package: mlflow - this notebook was tested with mlflow version 0.9.1
# MAGIC * Includes:
# MAGIC   * Utils_Python
# MAGIC * Synopsis:
# MAGIC   * Classifier: DecisionTreeRegressor
# MAGIC   * Logs  model in Spark ML and MLeap format
# MAGIC * Prediction: [Predict_PySpark_Model](https://demo.cloud.databricks.com/#notebook/2386138)
# MAGIC * Experiment: 2987091 -  [/Users/andre.mesarovic@databricks.com/gd_pyspark_Train_PySpark_Model_with_trackMLlib](https://demo.cloud.databricks.com/#mlflow/experiments/2987091)

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

spark.conf.set("spark.databricks.mlflow.trackMLlib.enabled",True)

# COMMAND ----------

# MAGIC %run ./UtilsTrain

# COMMAND ----------

metric_names = ["rmse","r2", "mae"]

dbutils.widgets.text(WIDGET_TRAIN_EXPERIMENT, create_experiment_name("pyspark_trackMLlib"))
dbutils.widgets.text("maxDepth", "0 2")
dbutils.widgets.text("maxBins", "32")
dbutils.widgets.text("numFolds", "2") # use 3+ folds in practice
dbutils.widgets.dropdown("metricName",metric_names[0],metric_names)
dbutils.widgets.dropdown("Show parent metrics","true",["false","true"])

experiment_name = dbutils.widgets.get(WIDGET_TRAIN_EXPERIMENT)
numFolds = int(dbutils.widgets.get("numFolds"))
metricName = dbutils.widgets.get("metricName")
maxDepthParams =  [int(x) for x in dbutils.widgets.get("maxDepth").split(" ")]
maxBinsParams =  [int(x) for x in dbutils.widgets.get("maxBins").split(" ")]
show_parent_metrics = dbutils.widgets.get("Show parent metrics") == "true"

experiment_name, maxDepthParams, maxBinsParams, numFolds, metricName, show_parent_metrics

# COMMAND ----------

if not experiment_name.isspace():
    experiment_id = set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data = read_wine_data()
(trainingData, testData) = data.randomSplit([0.7, 0.3], 2019)

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import mlflow.spark

# COMMAND ----------

def log_metric(predictions, metricName):
    evaluator = RegressionEvaluator(
        labelCol=colLabel, predictionCol=colPrediction, metricName=metricName)
    metricValue = evaluator.evaluate(predictions)
    mlflow.log_metric("_"+metricName, metricValue)
    print("  {}: {}".format(metricName,metricValue))
    return metricValue

# COMMAND ----------

run_name = "parentMetrics" if show_parent_metrics else ""
with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",experiment_id)
    
    # Log MLflow parameters
    print("Parameters:")
    print("  maxDepthParams:",maxDepthParams)
    print("  maxBinsParams:",maxBinsParams)

    # Create pipeline
    dt = DecisionTreeRegressor(labelCol=colLabel, featuresCol=colFeatures)
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=colFeatures)
    pipeline = Pipeline(stages=[assembler, dt])
    
    paramGrid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, maxDepthParams) \
        .addGrid(dt.maxBins, maxBinsParams) \
        .build()
        
    evaluator = RegressionEvaluator(
        labelCol=colLabel, predictionCol=colPrediction, metricName=metricName)

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=numFolds)  

    # Train model.  This also runs the indexers.
    cvModel = crossval.fit(trainingData)
    model = cvModel.bestModel

    # Make predictions.
    predictions = cvModel.transform(testData)
    predictions.select(colPrediction,colLabel,colFeatures).show(5)

    # Calculate some metrics for parent run
    if show_parent_metrics:
        print("Metrics:")
        for m in metric_names:
            log_metric(predictions, m)

    # Save models in Spark and MLeap foramt
    mlflow.spark.log_model(model, "spark-model")
    mlflow.mleap.log_model(spark_model=model, sample_input=testData, artifact_path="mleap-model")

    mlflow.set_tag("metricName",metricName)

# COMMAND ----------

# MAGIC %md #### Result

# COMMAND ----------

display_run_uri(experiment_id, run_id)

# COMMAND ----------

dbutils.notebook.exit(run_id+" "+experiment_id)
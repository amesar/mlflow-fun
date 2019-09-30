# Databricks notebook source
# MAGIC %md 
# MAGIC ## Distributed Hyperopt + Automated MLflow Tracking
# MAGIC 
# MAGIC Based upon Databricks public notebook: 
# MAGIC   * https://docs.databricks.com/_static/notebooks/hyperopt-spark-mlflow.html
# MAGIC   * https://docs.azuredatabricks.net/_static/notebooks/hyperopt-spark-mlflow.html
# MAGIC 
# MAGIC Databricks Runtime 5.4 ML and above offers an augmented version of [Hyperopt](https://github.com/hyperopt/hyperopt), a library for ML hyperparameter tuning in Python. It includes:
# MAGIC * Installed Hyperopt
# MAGIC * `SparkTrials` class, which provides distributed tuning via Apache Spark
# MAGIC * Automated MLflow tracking, configured via `spark.databricks.mlflow.trackHyperopt.enabled` (on by default)
# MAGIC 
# MAGIC #### Use case
# MAGIC Single-machine ML workloads in Python that you want to scale up and for which you want to track hyperparameter tuning.
# MAGIC 
# MAGIC #### In this example notebook
# MAGIC 
# MAGIC The demo is from Hyperopt's documentation with minor adjustments and can be swapped out for any other single-machine ML workload.

# COMMAND ----------

# MAGIC %md ### Single-machine Hyperopt workflow
# MAGIC 
# MAGIC First define a single-machine Hyperopt workflow, which has 3 main parts:
# MAGIC * Define a function to minimize
# MAGIC * Define a search space over hyperparameters
# MAGIC * Select a search algorithm
# MAGIC 
# MAGIC To get more familiar with Hyperopt APIs, check out the [Hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin).

# COMMAND ----------

# MAGIC %md
# MAGIC **Define a function to minimize**
# MAGIC 
# MAGIC * Inputs: hyperparameters
# MAGIC * Internally: compute loss with simple formulae
# MAGIC * Output: loss

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def train(params):
  """
  An example train method that computes the square of the input.
  This method will be passed to `hyperopt.fmin()`.
  
  :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
  :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
  """
  x, = params
  return {'loss': x ** 2, 'status': STATUS_OK}


# COMMAND ----------

# MAGIC %md 
# MAGIC **Define the search space over hyperparameters**
# MAGIC 
# MAGIC See the [Hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions) for details on defining a search space and parameter expressions.

# COMMAND ----------

search_space = [hp.uniform('x', -10, 10)]

# COMMAND ----------

# MAGIC %md
# MAGIC **Select a search algorithm**
# MAGIC 
# MAGIC The two main choices are:
# MAGIC * `hyperopt.tpe.suggest`: Tree of Parzen Estimators, a Bayesian approach which iteratively and adaptively selects new hyperparameter settings to explore based on past results
# MAGIC * `hyperopt.rand.suggest`: Random search, a non-adaptive approach which samples over the search space

# COMMAND ----------

algo=tpe.suggest

# COMMAND ----------

# MAGIC %md 
# MAGIC **Run model tuning with Hyperopt fmin()**
# MAGIC 
# MAGIC Set `max_evals` to the maximum number of points in hyperparameter space to test, that is, the maximum number of models to fit and evaluate.

# COMMAND ----------

argmin = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  max_evals=16)

# COMMAND ----------

argmin

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed tuning using Apache Spark and MLflow
# MAGIC 
# MAGIC To distribute tuning, you can add 1 more argument to `fmin()`, a `Trials` class called `SparkTrials`. `SparkTrials` takes 2 arguments:
# MAGIC * `parallelism`: Number of models to fit and evaluate in concurrently.
# MAGIC * `timeout`: Maximum time (in seconds) which fmin is allowed to take. This argument is optional.
# MAGIC 
# MAGIC Since the `train()` function runs quickly, the overhead of starting Spark jobs dominates, and using `SparkTrials` appears slower. In typical tuning, `train()` runs for a long time, and distributed tuning with `SparkTrails` can often achieve speedup over single-machine tuning.
# MAGIC 
# MAGIC Automated MLflow tracking is enabled by default. Call `mlflow.start_run()` before calling `fmin()` as shown in the example below.

# COMMAND ----------

from hyperopt import SparkTrials
help(SparkTrials)

# COMMAND ----------

spark_trials = SparkTrials(parallelism=2)

# COMMAND ----------

import mlflow

with mlflow.start_run():
  argmin = fmin(
    fn=train,
    space=search_space,
    algo=algo,
    max_evals=16,
    trials=spark_trials)

# COMMAND ----------

# MAGIC %md To view the MLflow experiment associated with the notebook, click the **Runs** icon in the notebook context bar on the upper right.  There, you can view all runs. You can also bring up the full MLflow UI by clicking the button on the upper right that reads **View Experiment UI** when you hover over it.
# MAGIC 
# MAGIC To understand the effect of tuning `x`:
# MAGIC 
# MAGIC 1. Select the resulting runs and click **Compare**.
# MAGIC 1. In the Scatter Plot, select **x** for X-axis and **loss** for Y-axis.
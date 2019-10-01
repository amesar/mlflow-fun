# Databricks notebook source
# MAGIC %md ## Train Keras/Tensorflow Model with MLflow
# MAGIC 
# MAGIC * Libraries:
# MAGIC   * PyPI package: mlflow
# MAGIC * Includes:
# MAGIC   * UtilsTrain
# MAGIC * Synopsis:
# MAGIC   * Keras + TensorFlow
# MAGIC   * Datasets are already mounted to /mnt/training from s3a://databricks-corp-training/common
# MAGIC   * Based on Brooke's Deep Learning [04 MLflow notebook](https://demo.cloud.databricks.com/#notebook/2923192) notebook
# MAGIC * Prediction: [05a_Keras_TF_Train](https://demo.cloud.databricks.com/#notebook/3538584)

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

import keras
print("Keras Version:", keras.__version__)
import tensorflow
print("TensorFlow Version:", tensorflow.__version__)

# COMMAND ----------

# MAGIC %run ./UtilsTrain

# COMMAND ----------

dbutils.widgets.text(WIDGET_TRAIN_EXPERIMENT, create_experiment_name("keras"))
dbutils.widgets.text("Epochs","10")
dbutils.widgets.dropdown("Optimizer","adam",["adam","sgd"])

experiment_name = dbutils.widgets.get(WIDGET_TRAIN_EXPERIMENT)
epochs = int(dbutils.widgets.get("Epochs"))
optimizer = dbutils.widgets.get("Optimizer")

epochs,optimizer, experiment_name

# COMMAND ----------

experiment_id = set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md ### Read Data

# COMMAND ----------

from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(42) # For reproducibility

cal_housing = fetch_california_housing()

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                        cal_housing.target,
                                                        test_size=0.2,
                                                        random_state=1)

print(cal_housing.DESCR)

# COMMAND ----------

# MAGIC %md
# MAGIC Build model architecture as before.

# COMMAND ----------

from keras.models import Sequential
from keras.layers import Dense

def build_model():
  return Sequential([Dense(20, input_dim=8, activation='relu'),
                    Dense(20, activation='relu'),
                    Dense(1, activation='linear')]) # Keep the last layer as linear because this is a regression problem

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train

# COMMAND ----------

# Note issue with **kwargs https://github.com/keras-team/keras/issues/9805
from mlflow.keras import log_model

def train(model, compile_kwargs, fit_kwargs, optional_params={}):
    '''
    This is a wrapper function for tracking expirements with MLflow
        
    Parameters
    ----------
    model: Keras model
        The model to track
        
    compile_kwargs: dict
        Keyword arguments to compile model with
    
    fit_kwargs: dict
        Keyword arguments to fit model with
    '''
    with mlflow.start_run() as run:
        mlflow.log_param("mlflow.version",mlflow.version.VERSION)
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        print("MLflow:")
        print("    run_id:",run_id)
        print("    experiment_id:",experiment_id)
        model = model()
        model.compile(**compile_kwargs)
        history = model.fit(**fit_kwargs)
        
        for param_key, param_value in {**compile_kwargs, **fit_kwargs, **optional_params}.items():
            if param_key not in ["x", "y", "X_val", "y_val"]:
                mlflow.log_param(param_key, param_value)
        
        for key, values in history.history.items():
            for v in values:
                    if not np.isnan(v): # MLflow won't log NaN
                        mlflow.log_metric(key, v)

        for i, layer in enumerate(model.layers):
            mlflow.log_param("hidden_layer_" + str(i) + "_units", layer.output_shape)
            
        log_model(model, "keras-model")
        return run_id 


# COMMAND ----------

# MAGIC %md
# MAGIC Setup arguments to training call

# COMMAND ----------

compile_kwargs = {
  "optimizer": "sgd", 
  "loss": "mse",
  "metrics": ["mse", "mae"],
}

fit_kwargs = {
  "x": X_train, 
  "y": y_train,
  "epochs": epochs,
  "verbose": 2
}

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's add some data normalization, as well as a validation dataset.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

fit_kwargs["validation_split"] = 0.2
fit_kwargs["x"] = X_train_scaled

optional_params = {
  "normalize_data": "true"
}

compile_kwargs["optimizer"] = optimizer
run_id = train(build_model, compile_kwargs, fit_kwargs, optional_params)

# COMMAND ----------

# MAGIC %md ### Result

# COMMAND ----------

display_run_uri(experiment_id, run_id)

# COMMAND ----------

dbutils.notebook.exit(run_id+" "+experiment_id)
# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import platform

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn
from wine_quality import plot_utils

print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

colLabel = "quality"

class Trainer(object):
    def __init__(self, experiment_name, data_path, run_origin="none"):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.run_origin = run_origin
        np.random.seed(40)

        print("experiment_name:",self.experiment_name)
        print("run_origin:",run_origin)

        # Read and prepare data
        print("data_path:",data_path)
        data = pd.read_csv(data_path)
        train, test = train_test_split(data)
    
        # The predicted column is "quality" which is a scalar from [3, 9]
        self.train_x = train.drop([colLabel], axis=1)
        self.test_x = test.drop([colLabel], axis=1)
        self.train_y = train[[colLabel]]
        self.test_y = test[[colLabel]]

        self.X = data.drop([colLabel], axis=1).values
        self.y = data[[colLabel]].values.ravel()

        # If using 'mlflow run' must use --experiment-id to set experiment since set_experiment() does not take effect
        if self.experiment_name != "none":
            mlflow.set_experiment(experiment_name)
            client = mlflow.tracking.MlflowClient()
            experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
            print("experiment_id:",experiment_id)


    def train(self, max_depth, max_leaf_nodes):
        with mlflow.start_run(run_name=self.run_origin) as run:  # NOTE: mlflow CLI ignores run_name
            run_id = run.info.run_uuid
            experiment_id = run.info.experiment_id
            print("MLflow:")
            print("  run_id:",run_id)
            print("  experiment_id:",experiment_id)

            # Create model
            dt = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
            print("Model:",dt)

            # Fit and predict
            dt.fit(self.train_x, self.train_y)
            predictions = dt.predict(self.test_x)

            # MLflow params
            print("Parameters:")
            print("  max_depth:",max_depth)
            print("  max_leaf_nodes:",max_leaf_nodes)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("max_leaf_nodes", max_leaf_nodes)

            # MLflow metrics
            rmse = np.sqrt(mean_squared_error(self.test_y, predictions))
            mae = mean_absolute_error(self.test_y, predictions)
            r2 = r2_score(self.test_y, predictions)
            print("Metrics:")
            print("  rmse:",rmse)
            print("  mae:",mae)
            print("  r2:",r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            
            # MLflow tags
            mlflow.set_tag("mlflow.runName",self.run_origin) # mlflow CLI picks this up
            mlflow.set_tag("data_path", self.data_path)
            mlflow.set_tag("exp_id", experiment_id)
            mlflow.set_tag("exp_name", self.experiment_name)
            mlflow.set_tag("run_origin", self.run_origin)
            mlflow.set_tag("platform", platform.system())
    
            # MLflow log model
            mlflow.sklearn.log_model(dt, "sklearn-model")
    
            # MLflow artifact - plot file
            plot_file = "plot.png"
            plot_utils.plot_me(self.test_y, predictions, plot_file)
            mlflow.log_artifact(plot_file)
    
        return (experiment_id,run_id)

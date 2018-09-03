# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import sys
sys.path.append("../..")

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, enet_path

import mlflow
import mlflow.sklearn
import mlflow_utils
import plot_utils

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train(data_path, alpha, l1_ratio):
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    #data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)
    print("data_path:",data_path)
    data = pd.read_csv(data_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    current_file = os.path.basename(__file__)
    experiment_name = "py/sk/ElasticNet/WineQuality"
    experiment_id = mlflow_utils.get_or_create_experiment_id(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, source_name=current_file):
        run_id = mlflow.active_run().info.run_uuid
        print("run_id:",run_id)
        clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        clf.fit(train_x, train_y)

        predicted_qualities = clf.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("data", data_path)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(clf, "model")

        eps = 5e-3  # the smaller it is the longer is the path
        X = data.drop(["quality"], axis=1).values
        y = data[["quality"]].values.ravel()

        alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=l1_ratio, fit_intercept=False)
        plot_file = "wine_ElasticNet-paths_{}_{}.png".format(alpha,l1_ratio)
        plot_utils.plot_enet_descent_path(X, y, l1_ratio, alphas_enet, coefs_enet, plot_file)
        mlflow.log_artifact(plot_file)


if __name__ == "__main__":
    if len(sys.argv) < 3: 
        print("ERROR: Expecting alpha and l1_ratio values")
        sys.exit(1)
    data_path = sys.argv[3] if len(sys.argv) > 3 else "wine-quality.csv"
    train(data_path, float(sys.argv[1]), float(sys.argv[2]))

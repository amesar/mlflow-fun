
# Serve predictions with mlflow.pyfunc.load_pyfunc()

from __future__ import print_function
import sys
import mlflow
import mlflow.pyfunc
import mlflow.tracking
import predict_util

if __name__ == "__main__":
    if len(sys.argv) < 1:
        println("ERROR: Expecting RUN_ID PREDICTION_FILE")
        sys.exit(1)
    print("MLflow Version:", mlflow.version.VERSION)
    run_id = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "../../data/wine-quality/wine-quality-red.csv"
    print("data_path:",data_path)
    print("run_id:",run_id)

    client = mlflow.tracking.MlflowClient()
    model_uri = client.get_run(run_id).info.artifact_uri + "/sklearn-model"
    print("model_uri:",model_uri)
    model = mlflow.pyfunc.load_pyfunc(model_uri)
    print("model:",model)
    predict_util.run_predictions(model,data_path)

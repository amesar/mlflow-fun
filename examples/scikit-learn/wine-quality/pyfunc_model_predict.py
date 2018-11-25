
# Serve predictions with mlflow.pyfunc.load_pyfunc()

from __future__ import print_function
import sys
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.tracking

if __name__ == "__main__":
    if len(sys.argv) < 1:
        println("ERROR: Expecting RUN_ID PREDICTION_FILE")
        sys.exit(1)
    print("MLflow Version:", mlflow.version.VERSION)
    run_id = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "wine-quality.csv"
    print("path:",path)
    print("run_id:",run_id)

    model_uri = mlflow.start_run(run_id).info.artifact_uri +  "/model"
    print("model_uri=",model_uri)
    model = mlflow.pyfunc.load_pyfunc(model_uri)
    print("model:",model)

    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_json(path)
    print("df.shape:",df.shape)
    print("df.columns:",df.columns)
    if 'quality' in df:
         df = df.drop(['quality'], axis=1)

    predictions = model.predict(df)
    print("predictions:",predictions)

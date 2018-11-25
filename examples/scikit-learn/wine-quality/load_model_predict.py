
# Serve predictions with mlflow.sklearn.load_model()

from __future__ import print_function
import sys
import pandas as pd
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    if len(sys.argv) < 1:
        println("ERROR: Expecting RUN_ID PREDICTION_FILE")
        sys.exit(1)
    print("MLflow Version:", mlflow.version.VERSION)
    run_id = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "wine-quality.csv"
    print("path:",path)
    print("run_id:",run_id)

    model = mlflow.sklearn.load_model("model", run_id=run_id)
    print("model:",model)

    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_json(path)

    print("df.shape:",df.shape)
    print("df.columns:",df.columns)
    if 'quality' in df:
         df = df.drop(['quality'], axis=1)
    predictions = model.predict(df)
    print("predictions:",predictions)

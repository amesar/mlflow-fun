
# Serve predictions from a saved MLflow model

import sys
import json
import pandas as pd
from pandas.io.json import json_normalize
import mlflow.sklearn
import mlflow

if __name__ == "__main__":
    if len(sys.argv) < 1:
        println("ERROR: Expecting RUN_ID PREDICTION_FILE")
        sys.exit(1)
    run_id = sys.argv[1] 
    data_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.json"
    print("run_id:",run_id)
    print("data_path:",data_path)
    clf = mlflow.sklearn.load_model("model",run_id=run_id)
    print("clf:",clf)
    with open(data_path, 'rb') as f:
        data = json.loads(f.read())
    df = json_normalize(data)
    predicted = clf.predict(df)
    print("predicted:",predicted)

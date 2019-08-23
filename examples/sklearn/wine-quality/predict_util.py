from __future__ import print_function
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def read_prediction_data(data_path):
    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_json(data_path, orient='split')
    if 'quality' in df:
        labels = df['quality']
        df = df.drop(['quality'], axis=1)
    else:
        labels = None
    return df,labels

def show_prediction_metrics(predictions, labels):
    rmse = np.sqrt(mean_squared_error(predictions, labels))
    mae = mean_absolute_error(predictions, labels)
    r2 = r2_score(predictions, labels)
    print("Metrics:")
    print("  rmse:",rmse)
    print("  mae: ",mae)
    print("  r2:  ",r2)

def run_predictions(model, data_path):
    df,labels = read_prediction_data(data_path)
    predictions = model.predict(df)
    print("predictions:",predictions)
    if labels is not None:
        show_prediction_metrics(predictions,labels)

'''MLFlow utilities'''

import mlflow

def get_or_create_experiment_id(experiment_name):
    tracking_uri = mlflow.tracking.get_tracking_uri()
    client = mlflow.tracking.MlflowClient(tracking_uri)

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = exp.experiment_id
    print("experiment_name={} experiment_id={}".format(experiment_name,experiment_id))
    return experiment_id

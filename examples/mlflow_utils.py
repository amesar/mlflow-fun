'''MLFlow utilities'''

import mlflow

def get_experiment_id(client, experiment_name):
    exps = client.list_experiments()
    for exp in exps:
        if experiment_name == exp.name:
            return exp.experiment_id
    return None

def get_or_create_experiment_id(experiment_name):
    tracking_uri = mlflow.tracking.get_tracking_uri()
    client = mlflow.tracking.MlflowClient(tracking_uri)

    experiment_id = get_experiment_id(client, experiment_name)
    if experiment_id == None:
        experiment_id = mlflow.create_experiment(experiment_name)
    print("experiment_id={} experiment_name={}".format(experiment_id,experiment_name))
    return experiment_id

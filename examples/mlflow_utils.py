# MLFlow utilities

import mlflow

def get_experiment_id(experiment_name):
    experiments = exps = mlflow.tracking.get_service().list_experiments()
    for exp in experiments:
        if experiment_name == exp.name:
            return exp.experiment_id
    return None

def get_or_create_experiment_id(experiment_name):
    experiment_id = get_experiment_id(experiment_name)
    if experiment_id == None:
        experiment_id = mlflow.create_experiment(experiment_name)
    print("experiment_name={} experiment_id={}".format(experiment_name,experiment_id))
    return experiment_id

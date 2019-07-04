
import mlflow
from mlflow_fun.metrics.api.best_run import get_best_run
from utils_test import create_experiment

def create_runs():
    exp = create_experiment()
    with mlflow.start_run() as run:
        mlflow.log_metric("m1", 0.1)
    run0 = run
    with mlflow.start_run() as run:
        mlflow.log_metric("m1", 0.2)
    return exp,run0,run

def test_descending():
    exp,run0,run1 = create_runs()
    best = get_best_run(exp.experiment_id, "m1", ascending=False)
    assert run1.info.run_id == best[0]

def test_ascending():
    exp,run0,run1 = create_runs()
    best = get_best_run(exp.experiment_id, "m1", ascending=True)
    assert run0.info.run_id == best[0]


import mlflow
from mlflow_fun.metrics.api.best_run import get_best_run
from utils_test import create_experiment

TAG_RUN_NAME = "mlflow.runName"
client = mlflow.tracking.MlflowClient()

def test_no_run_name():
    exp = create_experiment()
    with mlflow.start_run() as run:
        mlflow.log_metric("m1", 0.1)
    run2 = client.get_run(run.info.run_id)
    assert TAG_RUN_NAME not in run2.data.tags

def test_run_name():
    exp = create_experiment()
    with mlflow.start_run(run_name="foo") as run:
        mlflow.log_metric("m1", 0.1)
    run2 = client.get_run(run.info.run_id)
    assert "foo" == run2.data.tags[TAG_RUN_NAME]

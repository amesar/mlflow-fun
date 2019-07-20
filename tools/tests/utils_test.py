
import mlflow

client = mlflow.tracking.MlflowClient()
count = 0

def create_experiment():
    global count
    exp_name = "exp_"+str(count)
    count += 1
    mlflow.set_experiment(exp_name)
    return client.get_experiment_by_name(exp_name)

def create_runs():
    create_experiment()
    with mlflow.start_run() as run:
        mlflow.log_param("p1", "hi")
        mlflow.log_metric("m1", 0.786)
        mlflow.set_tag("t1", "hi")
    exp_id = run.info.experiment_id
    runs = client.search_runs([exp_id],"")
    return runs

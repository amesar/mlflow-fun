import sys
import mlflow

sys.stdout.write("MLflow Version: {}".format(mlflow.version.VERSION))
sys.stdout.write("MLflow Tracking URI: {}\n".format(mlflow.get_tracking_uri()))

def process(base_name, max_level, level, max_children, idx):
    if level >= max_level: 
        return
    name = "{}_{}_{}".format(base_name,level,idx)
    sys.stdout.write("name: {} max_level: {}\n".format(name,max_level))
    with mlflow.start_run(run_name=name, nested=(level > 0)) as run:
        sys.stdout.write("runId: {}\n".format(run.info.run_uuid))
        sys.stdout.write("experimentId: {}\n".format(run.info.experiment_id))
        mlflow.log_param("alpha", str(idx+0.1))
        mlflow.log_metric("auroch", 0.123)
        mlflow.set_tag("algo", name)
        with open("info.txt", "w") as f:
            f.write(name)
        mlflow.log_artifact("info.txt")
        for j in range(0, max_children):
            process(base_name, max_level, level+1, max_children, j)

if __name__ == "__main__":
    experiment_name = 'HelloWorld_NestedRuns'
    mlflow.set_experiment(experiment_name)
    max_children = 1 # if your increase this beware of exponential growth per max_level
    max_level = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    max_children = int(sys.argv[2]) if len(sys.argv) > 2 else max_children
    sys.stdout.write("max_level: {} max_children: {}\n".format(max_level,max_children))
    process('nst',max_level,0, max_children, 0)

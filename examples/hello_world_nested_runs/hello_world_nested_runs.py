import sys
import mlflow
from argparse import ArgumentParser

sys.stdout.write("MLflow Version: {}".format(mlflow.version.VERSION))
sys.stdout.write("MLflow Tracking URI: {}\n".format(mlflow.get_tracking_uri()))

def process(base_name, max_level, max_children, level=0, idx=0):
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
            process(base_name, max_level, max_children, level+1, j)

if __name__ == "__main__":
    experiment_name = 'HelloWorld_NestedRuns'
    mlflow.set_experiment(experiment_name)

    parser = ArgumentParser()
    parser.add_argument("--max_level", dest="max_level", help="Number of levels to recurse", default=1, type=int)
    parser.add_argument("--max_children", dest="max_children", help="Number of nodes at each level", default=1, type=int)
    # max_children: if your increase this beware of exponential growth per max_level
    args = parser.parse_args()

    sys.stdout.write("max_level: {} max_children: {}\n".format(args.max_level,args.max_children))
    process('nst',args.max_level,args.max_children)

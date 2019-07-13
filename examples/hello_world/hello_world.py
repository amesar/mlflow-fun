from __future__ import print_function
import os,sys
import mlflow

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME","hello_world")
print("experiment_name:",experiment_name)

def run(alpha, run_origin, log_artifact):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_origin) as run:  # NOTE: mlflow CLI ignores run_name
        print("runId:",run.info.run_uuid)
        print("artifact_uri:",mlflow.get_artifact_uri())
        print("alpha:",alpha)
        print("log_artifact:",log_artifact)
        print("run_origin:",run_origin)
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("rmse", 0.786)
        mlflow.set_tag("mlflow.runName",run_origin) # mlflow CLI picks this up
        mlflow.set_tag("run_origin", run_origin)
        mlflow.set_tag("log_artifact", log_artifact)
        if log_artifact:
            with open("info.txt", "w") as f:
                f.write("Hi artifact")
            mlflow.log_artifact("info.txt")

if __name__ == "__main__":
    alpha = sys.argv[1] if len(sys.argv) > 1 else "0.1"
    run_origin = sys.argv[2] if len(sys.argv) > 2 else "None"
    log_artifact = bool(sys.argv[3]) if len(sys.argv) > 3 else False
    run(alpha,run_origin,log_artifact)

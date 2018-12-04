from __future__ import print_function
import sys
import mlflow

def run(alpha, tag, log_artifact):
    with mlflow.start_run(run_name="Hello World") as run:
        print("runId:",run.info.run_uuid)
        print("artifact_uri:",mlflow.get_artifact_uri())
        print("alpha:",alpha)
        print("tag:",tag)
        print("log_artifact:",log_artifact)
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("auroch", 0.123)
        mlflow.set_tag("origin", tag)
        mlflow.set_tag("log_artifact", log_artifact)
        if log_artifact:
            with open("info.txt", "w") as f:
                f.write("Hi artifact")
            mlflow.log_artifact("info.txt")

if __name__ == "__main__":
    alpha = sys.argv[1] if len(sys.argv) > 1 else "0.1"
    tag = sys.argv[2] if len(sys.argv) > 2 else "None"
    log_artifact = bool(sys.argv[3]) if len(sys.argv) > 3 else False
    run(alpha,tag,log_artifact)

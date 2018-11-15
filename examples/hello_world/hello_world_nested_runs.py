from __future__ import print_function
import mlflow

experiment_id = 0
name = 'HelloWorld Nested Runs'

with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
    print("runId:",run.info.run_uuid," - name:",name)
    print("artifact_uri:",mlflow.get_artifact_uri())
    mlflow.set_tag("algo", name + " root")
    with open("info.txt", "w") as f:
        f.write("Hi root artifact")
    mlflow.log_artifact("info.txt")
    
    for i in range(0, 2):
        irun = str(i)
        name2 = name + " " + irun
        with mlflow.start_run(run_name=name2, nested=True) as run2:
            print("runId.2:",run2.info.run_uuid," - name2:",name2)
            mlflow.log_param("alpha", str(i+0.1))
            mlflow.log_metric("auroch", i+0.123)
            mlflow.set_tag("algo", "hello_world_"+irun)
            with open("info.txt", "w") as f:
                f.write("Hi artifact run "+irun)
            mlflow.log_artifact("info.txt")

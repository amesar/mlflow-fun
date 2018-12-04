from __future__ import print_function
import mlflow

experiment_id = 0
name = 'HelloWorld Nested Runs'

with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
    print("runId:",run.info.run_uuid," - name:",name)
    mlflow.set_tag("algo", name)
    with open("info.txt", "w") as f:
        f.write(name)
    mlflow.log_artifact("info.txt")
    for j in range(0, 2):
        name2 = name + " " + str(j)
        with mlflow.start_run(run_name=name2, nested=True) as run2:
            print("runId.2:",run2.info.run_uuid," - name2:",name2)
            mlflow.log_param("alpha", str(j+0.1))
            mlflow.log_metric("auroch", j+0.123)
            mlflow.set_tag("algo", name2)
            with open("info.txt", "w") as f:
                f.write(name2)
            mlflow.log_artifact("info.txt")

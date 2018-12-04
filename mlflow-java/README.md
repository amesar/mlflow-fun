# mlflow-java

MLflow Java and Scala extras

## Setup
You need to install Python MLflow Python in order for Java artifacts to work.
```
mvn -DskipTests=true package
pip install mlflow
```

## RunContext Fluent Proposal

* [RunContext.java](src/main/java/org/mlflow/tracking/RunContext.java)
* [HelloWorldFluent.scala](src/main/scala/org/mlflow/tracking/examples/HelloWorldFluent.scala) - Sample run with RunContext

Run:
```
scala -cp target/mlflow-fun-java-1.0-SNAPSHOT.jar \
  org.mlflow.tracking.examples.HelloWorldFluent \
  http://localhost:5000
```

## Some Tests

[src/test/scala/src/test/scala/org/mlflow](/src/test/scala/src/test/scala/org/mlflow/)

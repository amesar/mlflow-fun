# mlflow-java

MLflow Java and Scala extras.

## Setup
You need to install Python MLflow in order for Java artifacts to work.
```
mvn -DskipTests=true package
pip install mlflow
```

## RunContext Fluent Proposal

* [RunContext.java](src/main/java/org/mlflow/tracking/RunContext.java) - Implementation
* [HelloWorldFluent.scala](src/main/scala/org/mlflow/tracking/examples/HelloWorldFluent.scala) - Sample Scala run with RunContext
* [HelloWorldFluent.java](src/main/java/org/mlflow/tracking/examples/java2/HelloWorldFluent.java) - Sample Java run with RunContext


Run Scala example
```
scala -cp target/mlflow-fun-java-1.0-SNAPSHOT.jar \
  org.mlflow.tracking.examples.HelloWorldFluent \
  http://localhost:5000
```

Run Java example
```
java -classpath target/mlflow-fun-java-1.0-SNAPSHOT.jar \
  org.mlflow.tracking.examples.java2.HelloWorldFluent \
  http://localhost:5000 10
```

## Some Tests

[src/test/scala/src/test/scala/org/mlflow](src/test/scala/src/test/scala/org/mlflow)

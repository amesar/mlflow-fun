# mlflow-fun - Spark Scala Jackson Example

Scala Spark ML sample using older deprecated Jackson-based MLflow client.

See: [https://github.com/amesar/mlflow/tree/java-client-jackson/mlflow-java](https://github.com/amesar/mlflow/tree/java-client-jackson/mlflow-java).

**Install MLflow Java Client**

Until the Java client is pushed to Maven central, install it locally.

```
git clone -b java-client-jackson https://github.com/amesar/mlflow mlflow-java-client-jackson
cd mlflow-java-client-jackson
mvn -DskipTests=true install
```


**Run sample**

Source: [DecisionTreeRegressionExample.scala](src/main/scala/org/andre/mlflow/examples/DecisionTreeRegressionExample.scala)
```
cd examples/spark-scala-jackson
mvn package
spark-submit \
  --class org.andre.mlflow.examples.DecisionTreeRegressionExample \
  --master local[2] \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000 \
  data/sample_libsvm_data.txt

Experiment name: scala/DecisionTreeRegressionExample
Experiment ID: 2

```
**Check Results in UI**
```
http://localhost:5000/experiments/2
```

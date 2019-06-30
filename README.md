# mlflow-fun

Exploring MLflow in depth for Python and Scala.

## Examples
#### Python
* [Hello World](examples/hello_world) and [Hello World Nested Runs](examples/hello_world_nested_runs).
* [Python Scikit-learn](examples/sklearn/wine-quality/README.md) - most advanced example.
* [PySpark ML](examples/pyspark/README.md).
* [PyTorch ML](examples/pytorch/README.md).
* [Tools](tools) - Useful MLflow tools: dump run, dump experiment, dump runs to CSV files, etc.

#### Scala with Java client
* [Scala Spark ML examples](examples/scala/README.md) - uses MLFlow Java client.
* [Tools](examples/scala/README.md#Tools) - Useful MLflow tools: dump run, dump experiment, dump runs to CSV files, etc.
* Note: You must install Python MLflow for Java client to work: `pip install mlflow`.

#### Other
  * [mlflow-java](mlflow-java/README.md) - MLflow Java and Scala extras such as proposed [RunContext](mlflow-java/src/main/java/org/mlflow/tracking/RunContext.java).

## Setup

Before running the examples, you need to install the MLflow Python environment and launch an MLflow server.

### Install 

Install either with PyPi or Miniconda ([conda.yaml](conda.yaml)).

#### PyPi

```
pip install mlflow
```

#### Miniconda


* Install miniconda3: ``https://conda.io/miniconda.html``
* Create the environment: ``conda env create --file conda.yaml``
* Source the environment: `` source activate mlflow-fun``

### Run Server

```
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri $PWD/mlruns --default-artifact-root $PWD/mlruns
```

### Spark

For those examples that use Spark, download the latest Spark version to your local machine. See [Download Apache Spark](https://spark.apache.org/downloads.html).

### Databricks

To run the examples against a Databricks cluster see the following documentation:
* [Remote Execution on Databricks](https://mlflow.org/docs/latest/projects.html#remote-execution-on-databricks)
* [Databricks REST API](https://docs.databricks.com/api/latest/index.html) and 
[Databricks CLI](https://docs.databricks.com/user-guide/dev-tools/databricks-cli.html).

For examples see [Hello World](examples/hello_world) and [Scikit-learn Wine Quality](examples/sklearn/wine-quality).

Setup
```
export MLFLOW_TRACKING_URI=databricks
```
The token and tracking server URL will be picked up from your Databricks CLI default profile in `~/.databricks.cfg`.
You can also override these values with the following environment variables:
```
export DATABRICKS_TOKEN=MY_TOKEN
export DATABRICKS_HOST=https://myshard.cloud.databricks.com
```

## Legacy
  * [Scala Spark ML with deprecated Jackson-based MLflow client](examples/spark-scala-jackson/README.md)

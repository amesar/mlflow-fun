# mlflow-fun

MLflow examples for Python and Scala training models.

## Examples
* Python:
  * [Python Scikit-learn example](examples/scikit-learn/wine-quality/README.md)
  * [PySpark ML examples](examples/pyspark/README.md)
  * [PyTorch ML examples](examples/pytorch/README.md)
  * [Hello World example](examples/hello_world)
    * [Hello World Nested Runs example](examples/hello_world_nested_runs)
* Scala/Java:
  * Note: You need to install Python MLflow in order for Java artifacts to work: `pip install mlflow`
  * [Scala Spark ML examples](examples/spark-scala/README.md) - uses MLFlow Java client
  * [mlflow-java](mlflow-java/README.md) - MLflow Java and Scala extras such as proposed RunContext
* Legacy:
  * [Scala Spark ML with deprecated Jackson-based MLflow client](examples/spark-scala-jackson/README.md)


## Setup

Before running the examples, you need to install the MLflow Python environment and launch a MLflow server.

### Install 

Install either with PyPi or Miniconda ([conda.yaml](conda.yaml)).

#### PyPi

* Install MLflow from PyPi: ``pip install mlflow``

#### Miniconda


* Install miniconda3: ``https://conda.io/miniconda.html``
* Create the environment: ``conda env create --file conda.yaml``
* Source the environment: `` source activate mlflow-fun``

### Run Server

```
mlflow server --host 0.0.0.0 
```

### Spark

For those examples that use Spark, download the latest Spark version to your local machine. See [Download Apache Spark](https://spark.apache.org/downloads.html).

### Databricks

To run the examples against a Databricks cluster, see the 
[Databricks REST API](https://docs.databricks.com/api/latest/index.html) and 
[Databricks CLI](https://docs.databricks.com/user-guide/dev-tools/databricks-cli.html).
```
pip install databricks-cli 
```

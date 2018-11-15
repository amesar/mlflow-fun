# mlflow-fun

MLflow examples for Python and Scala training models.

## Examples
* Python examples:
  * [Python Scikit-learn examples](examples/scikit-learn/README.md)
  * [PySpark ML examples](examples/pyspark/README.md)
  * [PyTorch ML examples](examples/pytorch/README.md)
* [Scala Spark ML examples](examples/spark-scala/README.md) - uses MLFlow Java client
* Legacy:
  * [Scala Spark ML with deprecated Jackson-based MLflow client](examples/spark-scala-jackson/README.md)


## Install and Run MLflow Server

Before running the examples, you need to install the MLflow Python environment and launch a MLflow server.

### Install 

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

You are now ready to run the examples in the different languages.


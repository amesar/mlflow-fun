# mlflow-fun

MLflow examples for Python and Scala training models.

## Examples
* [Python Scikit-learn examples](examples/scikit-learn/README.md):
  *  Saves text and plot artifacts 
  *  Saves models in pickle format
  *  Serves models with mlflow.load_model() or MLflow serving web server
* [PySpark ML example](examples/pyspark/README.md)
* [Scala Spark ML example](examples/spark-scala/README.md) - uses MLFlow Java client
* [Scala Spark ML with deprecated Jackson-based MLflow client](examples/spark-scala-jackson/README.md)


## Install and Run MLflow Server

### Install mlflow either with PyPi or Miniconda.

Install mlflow either with PyPi or Miniconda.

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

You are now ready to run the examples various languages.


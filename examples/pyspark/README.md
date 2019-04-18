# mlflow-fun - PySpark example

## Overview

* Source: [train.py](train.py) and [predict.py](predict.py).
* Default experiment name: `py/spark/DecisionTree`
  * You can overwrite the experiment name with the environment variable MLFLOW_EXPERIMENT_NAME.

## Setup

* Install Spark on your machine.
* pip install mlflow


## Train

### Unmanaged without mlflow run

To run with standard main function
```
spark-submit --master local[2] train.py --max_depth 16 --max_bins 32
```

### Using mlflow run

These runs use the [MLproject](MLproject) file. For more details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Note that `mlflow run` ignores the `set_experiment()` function so you must specify the experiment with the  `--experiment-id` argument.

**mlflow run local**
```
mlflow run . -P max_depth=3 -P max_bins=24 --experiment-id=2019
```

**mlflow run github**
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/pyspark \
   -P max_depth=3 -P max_bins=24 \
  --experiment-id=2019
```

## Predict

See [predict.py](predict.py).

```
run_id=7b951173284249f7a3b27746450ac7b0
spark-submit --master local[2] predict.py $run_id
```

```
Predictions
root
 |-- label: double (nullable = true)
 |-- features: vector (nullable = true)
 |-- indexedLabel: double (nullable = false)
 |-- indexedFeatures: vector (nullable = true)
 |-- rawPrediction: vector (nullable = true)
 |-- probability: vector (nullable = true)
 |-- prediction: double (nullable = false)

+----------+------------+-----------+
|prediction|indexedLabel|probability|
+----------+------------+-----------+
|0.0       |1.0         |[1.0,0.0]  |
|1.0       |0.0         |[0.0,1.0]  |
|1.0       |0.0         |[0.0,1.0]  |
+----------+------------+-----------+
```


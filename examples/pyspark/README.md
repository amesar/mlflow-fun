# mlflow-fun - PySpark example

## Overview

* Data - Wine Quality
* Model - DecisionTreeRegressor
* Source: [train.py](train.py) and [predict.py](predict.py).
* Default experiment name: `pyspark`

## Setup

* Install Spark on your machine.
* pip install mlflow


## Train

### Unmanaged without mlflow run

To run with standard main function
```
spark-submit --master local[2] train.py --max_depth 16 --max_bins 32 \
  --data_path ../../examples/data/wine-quality/wine-quality-white.csv
```

### Using mlflow run

These runs use the [MLproject](MLproject) file. For more details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Note that `mlflow run` ignores the `set_experiment()` function so you must specify the experiment with the  `--experiment-id` argument.

**mlflow run local**
```
mlflow run .\
   -P max_depth=3 -P max_bins=24 \
   -P data_path ../../examples/data/wine-quality/wine-quality-white.csv \
   --experiment-id=2019
```

**mlflow run github**
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/pyspark \
   -P max_depth=3 -P max_bins=24 \
   -P data_path ../../examples/data/wine-quality/wine-quality-white.csv \
  --experiment-id=2019
```

## Predict

See [predict.py](predict.py).

```
run_id=7b951173284249f7a3b27746450ac7b0
spark-submit --master local[2] predict.py --run_id $run_id \
  --data_path ../../examples/data/wine-quality/wine-quality-white.csv
```

```
Predictions:
+----------------+-------+--------------------------------------------------------+
|prediction      |quality|features                                                |
+----------------+-------+--------------------------------------------------------+
|5.88032931490738|6      |[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]|
|5.88032931490738|6      |[6.3,0.3,0.34,1.6,0.049,14.0,132.0,0.994,3.3,0.49,9.5]  |
|5.88032931490738|6      |[8.1,0.28,0.4,6.9,0.05,30.0,97.0,0.9951,3.26,0.44,10.1] |
|5.88032931490738|6      |[7.2,0.23,0.32,8.5,0.058,47.0,186.0,0.9956,3.19,0.4,9.9]|
|5.88032931490738|6      |[7.2,0.23,0.32,8.5,0.058,47.0,186.0,0.9956,3.19,0.4,9.9]|
+----------------+-------+--------------------------------------------------------+
```

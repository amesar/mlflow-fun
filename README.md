# mlflow-fun

MLflow samples.

## Install

Install mlflow in either of two ways.

### 1. PyPi

Install MLflow from PyPi via ``pip install mlflow``

### 2. or miniconda

**Install miniconda3**
```
https://conda.io/miniconda.html
```

**Create environment**
```
conda env create --file conda.yaml
```
**Source environment**
```
source activate mlflow-fun
```

## Run 

**Launch server**
```
mlflow server --host 0.0.0.0 
```
**Run sample**
```
export MLFLOW_TRACKING_URI=http://localhost:5000
cd examples/sklearn
python iris_decision_tree.py
```
**Check Results in UI**
```
http://localhost:5000/experiments/1
```

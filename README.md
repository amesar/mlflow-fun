# MLflow Samples

## Install
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
cd sklearn
python sklearn_sample.py
```
**Check Results in UI**
```
http://localhost:5000/experiments/1
```

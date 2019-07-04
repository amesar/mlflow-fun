
import mlflow
from mlflow_fun.common.runs_to_pandas_converter import RunsToPandasConverter
from utils_test import create_runs

client = mlflow.tracking.MlflowClient()

def test_basic():
    exp_name = "runs_to_pandas"
    mlflow.set_experiment(exp_name)
    with mlflow.start_run() as run:
        mlflow.log_param("p1", "hi")
        mlflow.log_metric("m1", 0.786)
        mlflow.set_tag("t1", "hi")
    with mlflow.start_run() as run:
        mlflow.log_param("p2", "hi")
        mlflow.log_metric("m2", 0.786)
        mlflow.set_tag("t2", "hi")

    exp_id = run.info.experiment_id
    runs = client.search_runs([exp_id],"")

    converter = RunsToPandasConverter()
    df = converter.to_pandas_df(runs)
    assert df.shape[0] == len(runs)
    columns = set(df.columns)
    assert "_p_p1" in columns
    assert "_m_m1" in columns
    assert "_t_t1" in columns

def test_sort():
    runs = create_runs()
    converter = RunsToPandasConverter(do_sort=True)
    df = converter.to_pandas_df(runs)
    columns = list(df.columns)
    assert columns[0] == "run_id"
    columns.index("_p_p1") < columns.index("_m_m1")
    columns.index("_m_m1") < columns.index("_t_t1")

def test_skip_params():
    runs = create_runs()
    converter = RunsToPandasConverter(skip_params=True)
    df = converter.to_pandas_df(runs)
    assert not "_p_p1" in df.columns
    assert "_m_m1" in df.columns
    assert "_t_t1" in df.columns

def test_skip_metrics():
    runs = create_runs()
    converter = RunsToPandasConverter(skip_metrics=True)
    df = converter.to_pandas_df(runs)
    assert "_p_p1" in df.columns
    assert not "_m_m1" in df.columns
    assert "_t_t1" in df.columns

def test_skip_tags():
    runs = create_runs()
    converter = RunsToPandasConverter(skip_tags=True)
    df = converter.to_pandas_df(runs)
    assert "_p_p1" in df.columns
    assert "_m_m1" in df.columns
    assert not "_t_t1" in df.columns

def test_duration():
    runs = create_runs()
    converter = RunsToPandasConverter(do_duration=True)
    df = converter.to_pandas_df(runs)
    assert "__duration" in df.columns
    converter = RunsToPandasConverter()
    df = converter.to_pandas_df(runs)
    assert not "__duration" in df.columns

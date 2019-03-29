
from mlflow_fun.metrics.dataframe_builder import get_data_frame_builder

def get_best(experiment_id, metric, ascending=True, which="fast"):
    if not metric.startswith("_m_"): 
        metric = "_m_" + metric
    builder = get_data_frame_builder(which)
    df = builder.build_dataframe(experiment_id)
    df = df.select("run_uuid", metric).filter("{} is not NULL".format(metric)).sort(metric,ascending=ascending)
    return df.head()

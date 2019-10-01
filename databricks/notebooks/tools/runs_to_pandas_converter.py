# Databricks notebook source
# MAGIC %md runs_to_pandas_converter.py and sparse_utils.py
# MAGIC * Note: deprecated since MLflow 1.1.0 now provides this feature
# MAGIC * https://github.com/amesar/mlflow-fun/blob/master/tools/mlflow_fun/common/runs_to_pandas_converter.py
# MAGIC * https://github.com/amesar/mlflow-fun/blob/master/tools/mlflow_fun/common/sparse_utils.py
# MAGIC * synchronized 2019-07-03

# COMMAND ----------

import pandas as pd

"""
Sparse table utilities. 
Needs more research as to Python best practices for large datasets.
"""

""" Return a list of sparse dicts derived from a list of heterogenous dicts. """
def to_sparse_list_of_dicts(list_of_dicts):
    keys = create_keys(list_of_dicts)
    return [ { key: dct.get(key,None) for key in keys } for dct in list_of_dicts ]

""" Return a list of sparse lists derived from a list of heterogenous dicts. """
def to_sparse_list_of_lists(list_of_dicts):
    keys = list(create_keys(list_of_dicts))
    keys.sort()
    return (keys, [ [ dct.get(key,None) for key in keys ] for dct in list_of_dicts])

""" Create canonical set of keys by union-ing all keys of dicts """
def create_keys(list_of_dicts):
    keys = set()
    for dct in list_of_dicts:
        keys.update(set(dct.keys()))
    return keys

""" Return a sparse Pandas dataframe from a list of heterogenous dicts. """
def to_sparse_pandas_df(list_of_dicts):
    return pd.DataFrame.from_dict(v)

# COMMAND ----------

import pandas as pd
#from mlflow_fun.common import sparse_utils

''' 
Flatten all runs as a sparse dict and create Pandas dataframe.
For each run, all fields of info, data.params, data.metrics and data.tags are flattened into one sparse dict.
Parameters are prefixed with _p_, metrics with _m_ and tags with _t_.
Example: { "_p_alpha": "0.1", "_m_rmse": 0.82, "_t_data_path": "data.csv", "run_id: "123" }
'''
class RunsToPandasConverter(object):
    def __init__(self, do_sort=False, do_pretty_time=False, do_duration=False, skip_params=False, skip_metrics=False, skip_tags=False, nan_to_blank=False):
        self.do_sort = do_sort
        self.do_pretty_time = do_pretty_time
        self.do_duration = do_duration
        self.skip_params = skip_params
        self.skip_metrics = skip_metrics
        self.skip_tags = skip_tags
        self.nan_to_blank = nan_to_blank
        pass

    def _to_list_of_dicts(self, runs):
        rows = []
        for run in runs:
            dct = run.to_dictionary()
            dct = dct['info']
            if 'run_id' in dct: # in 1.0.0 both appear though only run_id is documented
                dct.pop('run_uuid', None)
            if not self.skip_params: self._merge(dct, run.data.params, '_p_')
            if not self.skip_metrics: self._merge(dct, run.data.metrics, '_m_')
            if not self.skip_tags: self._merge(dct, run.data.tags, '_t_')
            info = dct.pop('info', None)
            dct.pop('data', None)
            if self.do_duration:
                dur = dct['end_time'] - dct['start_time']
                if self.do_pretty_time:
                    dur = float(dur)/1000
                dct['__duration'] = dur
            rows.append(dct)
        return rows

    # Return a sparse list of dicts for all run data.
    def to_sparse_list_of_dicts(self, runs):
        runs =  self._to_list_of_dicts(runs)
        return to_sparse_list_of_dicts(runs)
        ##return sparse_utils.to_sparse_list_of_dicts(runs)

    # Return a sparse list of list for all run data.
    def to_sparse_list_of_lists(self, runs):
        runs =  self._to_list_of_dicts(runs)
        return to_sparse_list_of_lists(runs)
        ##return sparse_utils.to_sparse_list_of_lists(runs)

    # Return a sparse pandas dataframe for all run data.
    def to_pandas_df(self, runs):
        runs = self.to_sparse_list_of_dicts(runs)
        df = pd.DataFrame.from_dict(runs)
        if self.do_sort:
            df = df[self._sort_columns(df)]
        if self.nan_to_blank:
            df = df.replace(pd.np.nan, '', regex=True)
        if self.do_pretty_time:
            columns = list(df.columns)
            self.format_time(df, 'start_time')
            if 'end_time' in columns:
                 self.format_time(df, 'end_time')
        return df


    def _sort_columns(self, df):
        ARTIFACT_URI = "artifact_uri"
        skipme_first = [ "run_id", "start_time", "end_time" ]
        skipme_all = skipme_first + [ARTIFACT_URI]
        columns = list(df.columns)
        i,p,m,t = [],[],[],[]
        for c in columns:
            if c.startswith("_m_"): m.append(c)
            elif c.startswith("_p_"): p.append(c)
            elif c.startswith("_t_"): t.append(c)
            elif c not in skipme_all: i.append(c)
        columns = skipme_first + i + [ARTIFACT_URI] + sorted(p) + sorted(m) + sorted(t)
        return columns

    def _merge(self, row_all, row_new, prefix):
        row = { prefix+k:v for k,v in row_new.items() }
        row_all.update(row)

    def _strip_underscores(self, obj):
        return { k[1:]:v for (k,v) in obj.__dict__.items() }

    def format_time(self, df, column):
        df[column] = pd.to_datetime(df[column], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
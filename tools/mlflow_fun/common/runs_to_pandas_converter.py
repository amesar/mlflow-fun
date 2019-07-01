import pandas as pd
from mlflow_fun.common import sparse_utils
from mlflow_fun.common import mlflow_utils

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
        return sparse_utils.to_sparse_list_of_dicts(runs)

    # Return a sparse list of list for all run data.
    def to_sparse_list_of_lists(self, runs):
        runs =  self._to_list_of_dicts(runs)
        return sparse_utils.to_sparse_list_of_lists(runs)

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

if __name__ == "__main__":
    from argparse import ArgumentParser
    from mlflow_fun.common.mlflow_smart_client import MlflowSmartClient
    import mlflow
    print("MLflow Version:", mlflow.version.VERSION)

    parser = ArgumentParser()
    parser.add_argument("--experiment", dest="experiment_id_or_name", help="Experiment ID", required=True)
    parser.add_argument("--artifact_max_level", dest="artifact_max_level", help="Number of artifact levels to recurse", required=False, default=1, type=int)
    parser.add_argument("--sort", dest="sort", help="Show run info", required=False, default=False, action='store_true')
    parser.add_argument("--pretty_time", dest="pretty_time", help="Show info", required=False, default=False, action='store_true')
    parser.add_argument("--duration", dest="duration", help="Show duration", required=False, default=False, action='store_true')
    parser.add_argument("--nan_to_blank", dest="nan_to_blank", help="nan_to_blank", required=False, default=False, action='store_true')
    parser.add_argument("--skip_params", dest="skip_params", help="skip_params", required=False, default=False, action='store_true')
    parser.add_argument("--skip_metrics", dest="skip_metrics", help="skip_metrics", required=False, default=False, action='store_true')
    parser.add_argument("--skip_tags", dest="skip_tags", help="skip_tags", required=False, default=False, action='store_true')
    parser.add_argument("--csv_file", dest="csv_file", help="CSV file")

    args = parser.parse_args()
    print("Options:")
    for arg in vars(args):
        print("  {}: {}".format(arg,getattr(args, arg)))

    client = mlflow.tracking.MlflowClient()
    smart_client = MlflowSmartClient()
    exp = mlflow_utils.get_experiment(client, args.experiment_id_or_name)
    exp_id = exp.experiment_id
    print("experiment_id:",exp_id)
    runs = smart_client.list_runs(exp_id)

    converter = RunsToPandasConverter(args.sort, args.pretty_time, args.duration, args.skip_params, args.skip_metrics, args.skip_tags)

    df = converter.to_pandas_df(runs)
    path = "exp_runs_{}.csv".format(exp_id) if args.csv_file is None else args.csv_file
    print("Output CSV file:",path)
    with open(path, 'w') as f:
        df.to_csv(f, index=False)

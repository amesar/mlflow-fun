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


import mlflow
from mlflow_fun.common import sparse_utils 

def test_to_sparse_list_of_dicts():
    data = [ { "k0": "v0", "k1": "v1" }, { "k0": "v0", "k2": "v2" }]
    lst = sparse_utils.to_sparse_list_of_dicts(data)
    assert len(lst) == 2
    for dct in lst:
        assert len(dct) == 3
        assert dct.keys() == set(["k0", "k1", "k2"])
    assert lst[0]["k2"] == None
    assert lst[1]["k1"] == None

def test_to_sparse_list_of_lists():
    data = [ { "k0": "v0", "k1": "v1" }, { "k0": "v0", "k2": "v2" }]
    keys,lst = sparse_utils.to_sparse_list_of_lists(data)
    assert len(lst) == 2
    assert len(keys) == 3
    assert keys == ["k0", "k1", "k2"]
    for lst2 in lst:
        assert len(lst2) == 3
    assert lst[0][0] == "v0"
    assert lst[0][1] == "v1"
    assert lst[0][2] == None
    assert lst[1][0] == "v0"
    assert lst[1][1] == None
    assert lst[1][2] == "v2"

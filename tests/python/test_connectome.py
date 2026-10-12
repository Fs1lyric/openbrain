import numpy as np
import pytest

from openbrain.connectome import load_edges


def test_loads_edge_list_into_post_by_pre_matrix(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text("pre,post,weight\nAVAL,AVAR,3\nAVAR,PVCL,1.5\nAVAL,AVAR,2\n")
    names, weights = load_edges(path)
    assert names == ["AVAL", "AVAR", "PVCL"]
    i = {n: k for k, n in enumerate(names)}
    assert weights[i["AVAR"], i["AVAL"]] == 5.0  # duplicate edges sum
    assert weights[i["PVCL"], i["AVAR"]] == 1.5
    assert np.count_nonzero(weights) == 2


def test_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("from,to\nA,B\n")
    with pytest.raises(ValueError, match="pre, post, weight"):
        load_edges(path)

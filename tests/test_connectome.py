import numpy as np

from flycraft.connectome import Connectome


def test_from_edges_merges_duplicates_and_drops_zero():
    c = Connectome.from_edges([0, 0, 1, 2], [1, 1, 2, 0], [2.0, 3.0, -1.0, 0.0], np.arange(3))
    d = c.to_dense()
    assert d[0, 1] == 5.0
    assert d[1, 2] == -1.0
    assert d[2, 0] == 0.0
    assert c.n_connections == 2
    assert c.n_synapses == 6


def test_select_and_roundtrip(tmp_path):
    ann = {
        "cell_type": np.array(["A", "B", "A"]),
        "side": np.array(["left", "left", "right"]),
        "super_class": np.array(["central", "sensory", "central"]),
    }
    c = Connectome.from_edges([0], [1], [1.0], np.array([11, 22, 33]), ann=ann, name="t")
    assert c.select(cell_type="A").tolist() == [0, 2]
    assert c.select(cell_type="A", side="right").tolist() == [2]
    assert c.select(cell_type=["A", "B"], side="left").tolist() == [0, 1]
    assert c.index_of_root([33, 99]).tolist() == [2]
    path = tmp_path / "c.npz"
    c.save(path)
    c2 = Connectome.load(path)
    assert c2.name == "t"
    assert np.array_equal(c2.indices, c.indices)
    assert c2.select(cell_type="A").tolist() == [0, 2]

"""本物の FlyWire データがある場合だけ走るテスト（`python -m flycraft download` 後）。"""

import numpy as np
import pytest

from flycraft import data
from flycraft.brain import Brain
from flycraft.motor import MN9_ROOT_IDS

pytestmark = pytest.mark.skipif(not data.has_flywire(), reason="FlyWire データが無い")


@pytest.fixture(scope="module")
def con():
    return data.load_flywire()


def test_sugar_neurons_drive_proboscis_motor_neuron(con):
    """Shiu et al. 2024 の主結果: 糖受容ニューロンの活性化で MN9 が発火する。"""
    sugar = con.select(cell_sub_class="sugar/water")
    mn9 = con.index_of_root(MN9_ROOT_IDS)
    assert len(sugar) > 100 and len(mn9) == 2
    b = Brain(con, seed=0)
    b.set_drive(sugar, np.full(len(sugar), 150.0))
    counts = b.run(1000.0)
    assert counts[mn9].sum() > 10
    b = Brain(con, seed=0)
    assert b.run(300.0)[mn9].sum() == 0  # 入力なしでは発火しない


def test_looming_neurons_drive_giant_fiber(con):
    """LPLC2（ルーミング検出）→ 巨大繊維 DNp01（逃避）。"""
    lplc2 = con.select(cell_type="LPLC2", side="left")
    gf = con.select(cell_type="DNp01", side="left")
    b = Brain(con, seed=0)
    b.set_drive(lplc2, np.full(len(lplc2), 100.0))
    assert b.run(500.0)[gf].sum() > 3


def test_retinotopy_dorsal_rim_is_up(con):
    from flycraft.retinotopy import photoreceptor_directions

    idx, phi, theta = photoreceptor_directions(con)
    dra = np.isin(idx, con.select(cell_sub_class="DRA"))
    assert theta[dra].mean() > 30
    left = con.ann["side"][idx] == "left"
    assert phi[left].mean() < 0 < phi[~left].mean()

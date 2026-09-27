import math

import numpy as np
import pytest

from flycraft.brain import HAVE_NUMBA, Brain, LIFParams
from flycraft.connectome import Connectome


def random_net(n=400, m=6000, seed=0):
    rng = np.random.default_rng(seed)
    return Connectome.from_edges(rng.integers(0, n, m), rng.integers(0, n, m),
                                 rng.integers(-3, 14, m).astype(np.float32), np.arange(n))


def test_single_synapse_psp_matches_exact_solution():
    """1 本のシナプスの PSP が線形 ODE の厳密解と一致する。"""
    con = Connectome.from_edges([0], [1], [10.0], np.arange(2))
    p = LIFParams(std_U=0.0)
    b = Brain(con, p, engine="numpy")
    b.v[0] = -40.0  # 最初のステップで 0 番を発火させる
    trace = []
    for _ in range(60):
        b.run(p.dt)
        trace.append(float(b.v[1]))
    g0 = 10.0 * p.w_syn
    t_arrive = (1 + p.delay_steps) * p.dt  # 発火ステップの次から遅延後に g に加算される
    ts = np.arange(1, 61) * p.dt
    k = p.tau_syn / (p.tau_syn - p.tau_m)
    expect = np.where(ts >= t_arrive,
                      p.v0 + g0 * k * (np.exp(-(ts - t_arrive + p.dt) / p.tau_syn)
                                       - np.exp(-(ts - t_arrive + p.dt) / p.tau_m)), p.v0)
    assert np.allclose(trace, expect, atol=0.02)
    assert max(trace) > p.v0 + 0.3  # 10 シナプス ≈ 0.4 mV の EPSP


def test_bias_current_rate_matches_theory():
    con = Connectome.from_edges([], [], [], np.arange(1))
    for mv in (8.0, 10.0, 12.0):
        b = Brain(con, LIFParams(), engine="numpy")
        b.set_bias([0], mv)
        rate = b.run(2000.0)[0] / 2.0
        theory = 1000.0 / (-20.0 * math.log(1 - 7.0 / mv) + 2.2)
        assert abs(rate - theory) / theory < 0.1
    b = Brain(con, LIFParams(), engine="numpy")
    b.set_bias([0], 6.5)  # 閾値未満
    assert b.run(500.0)[0] == 0


def test_poisson_drive_rate():
    con = Connectome.from_edges([], [], [], np.arange(50))
    b = Brain(con, LIFParams(), engine="numpy", seed=1)
    b.set_drive(np.arange(50), np.full(50, 40.0))
    rate = b.run(2000.0).mean() / 2.0
    assert 34 < rate < 46


def test_short_term_depression_reduces_transmission():
    con = Connectome.from_edges([0], [1], [40.0], np.arange(2))
    out = {}
    for u in (0.0, 0.5):
        b = Brain(con, LIFParams(std_U=u), engine="numpy", seed=0)
        b.con.ann["super_class"][:] = "central"
        b.std_U[:] = u
        b.set_drive([0], [200.0])
        out[u] = b.run(1000.0)[1]
    assert out[0.5] < out[0.0]


def test_silence_blocks_spikes():
    con = random_net()
    b = Brain(con, engine="numpy", seed=0)
    b.set_drive(np.arange(20), np.full(20, 100.0))
    b.silence(np.arange(20))
    assert b.run(100.0)[:20].sum() == 0


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba が無い")
def test_numba_and_numpy_engines_are_identical():
    con = random_net(n=3000, m=40000)
    bs = [Brain(con, engine=e, seed=3) for e in ("numba", "numpy")]
    for b in bs:
        b.v[:10] = -40.0
        b.set_drive(np.arange(40), np.full(40, 60.0))
        b.set_bias(np.arange(100, 105), 9.0)
    for _ in range(30):
        c0, c1 = bs[0].run(5.0), bs[1].run(5.0)
        assert np.array_equal(c0, c1)
        assert np.allclose(bs[0].v, bs[1].v, atol=1e-3)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba が無い")
def test_lazy_engine_matches_reference():
    """遅延評価版（入力が来たニューロンと閾値に届きうるものだけ積分）も同じスパイクを出す。"""
    con = random_net(n=3000, m=40000)
    bs = [Brain(con, engine=e, seed=5) for e in ("numba-lazy", "numpy")]
    for b in bs:
        b.v[:10] = -40.0
        b.set_drive(np.arange(40), np.full(40, 60.0))
        b.set_bias(np.arange(100, 105), 9.0)
    for _ in range(40):
        assert np.array_equal(bs[0].run(5.0), bs[1].run(5.0))
    assert np.allclose(bs[0].v, bs[1].v, atol=0.1)


def test_thread_tuner_picks_fastest():
    from flycraft.brain import ThreadTuner

    t = ThreadTuner(8, trial_ms=100.0, revisit_ms=1e9)
    cost = {1: 1.6, 2: 1.0, 3: 1.2, 4: 1.3, 6: 2.0, 8: 7.0}
    th = t.current
    for _ in range(40):
        th = t.record(cost[th] * 0.05, 50.0)
    assert th == 2

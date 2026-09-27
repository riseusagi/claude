import numpy as np

from flycraft.backends import sim
from flycraft.backends.sim import AIR, FLOWER, SimWorld, _render_py
from flycraft.interface import Action


def test_world_is_deterministic_and_renders():
    a, b = SimWorld(seed=3), SimWorld(seed=3)
    assert np.array_equal(a.world, b.world)
    o = a.reset()
    assert o.frame.shape == (a.height, a.width, 3)
    assert o.frame.dtype == np.uint8
    assert o.frame.std() > 5  # 空一色ではない


def test_gravity_and_ground():
    w = SimWorld(seed=0)
    w.reset()
    p = w.player
    p.y += 3.0
    for _ in range(40):
        w.step(Action(), 0.05)
    assert p.on_ground
    assert not w._collides(p.x, p.y, p.z)
    assert w._collides(p.x, p.y - 0.1, p.z)


def test_walking_moves_player_forward():
    w = SimWorld(seed=0, n_slimes=0)
    w.reset()
    p = w.player
    fx, fz = w._forward_vec()
    x0, z0 = p.x, p.z
    for _ in range(10):
        w.step(Action(forward=1.0), 0.05)
    moved = (p.x - x0) * fx + (p.z - z0) * fz
    assert moved > 0.5 or w.stats["bumps"] > 0


def test_turning_changes_yaw_left():
    w = SimWorld(seed=0, n_slimes=0)
    w.reset()
    y0 = w.player.yaw
    w.step(Action(turn=1.0), 0.1)
    assert abs(((w.player.yaw - y0 + 180) % 360) - 180 - 18.0) < 1e-6


def test_flower_gives_sugar():
    w = SimWorld(seed=0, n_slimes=0)
    w.reset()
    p = w.player
    x, y, z = int(p.x), int(p.y), int(p.z)
    w.world[x, y, z] = FLOWER
    o = w.step(Action(), 0.05)
    assert o.sugar > 0.5
    assert w.world[x, y, z] == AIR
    assert w.stats["flowers"] == 1


def test_numpy_renderer_matches_numba():
    w = SimWorld(seed=1)
    w.reset()
    p = w.player
    d = w._camera_dirs()
    ref = _render_py(w.world, p.x, p.y + 1.62, p.z, d, 48.0, None, None, None)
    if sim.HAVE_NUMBA:
        got = sim._render_nb(w.world, p.x, p.y + 1.62, p.z, d, 48.0)
        assert np.mean(ref[1] == got[1]) > 0.999

import numpy as np
import pytest

from flycraft.backends.sim import SimWorld
from flycraft.fly import Fly, FlyConfig
from flycraft.interface import Action, Observation
from flycraft.retinotopy import infer
from flycraft.runner import Session
from flycraft.toy import build_toy


@pytest.fixture(scope="module")
def toy():
    con = build_toy()
    return con, infer(con)


def make(toy, **kw):
    con, r = toy
    return Fly(con, r, FlyConfig(seed=0, **kw))


def run(fly, obs, steps=20):
    for _ in range(steps):
        a = fly.step(obs, 50.0)
    return a


def test_sugar_triggers_biting(toy):
    fly = make(toy, hunger=0.0)
    a = run(fly, Observation(sugar=1.0))
    assert a.attack
    assert fly.motor.rates["feed_both"] > 4


def test_hunger_drives_walking(toy):
    hungry = run(make(toy, hunger=1.0), Observation())
    full = run(make(toy, hunger=0.0), Observation())
    assert hungry.forward > 0.3
    assert full.forward < hungry.forward


def test_optogenetic_turning(toy):
    fly = make(toy, hunger=0.0)
    fly.opto["dna02_l"] = True
    assert run(fly, Observation()).turn > 0.3
    fly = make(toy, hunger=0.0)
    fly.opto["dna02_r"] = True
    assert run(fly, Observation()).turn < -0.3


def test_giant_fiber_makes_jump(toy):
    fly = make(toy, hunger=0.0)
    fly.opto["gf"] = True
    jumps = sum(fly.step(Observation(), 50.0).jump for _ in range(10))
    assert jumps >= 5


def test_obstacle_reflex_turns_away(toy):
    fly = make(toy, hunger=1.0)
    turns = [fly.step(Observation(touch_left=1.0), 50.0).turn for _ in range(30)]
    assert fly.reflex.count >= 1
    assert min(turns) < -0.5  # 左に触れたら右へ


def test_session_in_sim_world(toy):
    con, r = toy
    fly = Fly(con, r, FlyConfig(seed=1))
    world = SimWorld(seed=1, width=96, height=54)
    s = Session(world, fly, dt_ms=50.0, realtime=False)
    for _ in range(60):
        s.step_once()
    s._publish()
    snap = s.snapshot()
    assert snap["tick"] == 60
    assert "eye" in snap and "frame" in snap and "act_idx" in snap
    assert fly.brain.t_ms == pytest.approx(3000.0)
    assert world.stats["distance"] > 0.5

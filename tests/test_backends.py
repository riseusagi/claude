import json
import socket
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from flycraft.backends.bedrock_ws import BedrockWSBackend, parse_querytarget
from flycraft.interface import Action

sys.path.insert(0, str(Path(__file__).parent))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_parse_querytarget():
    body = {"statusCode": 0, "details": json.dumps([{"dimension": 0, "position": {"x": 1.5, "y": 70.62, "z": -3},
                                                     "uniqueId": "-1", "yRot": 45.0}])}
    q = parse_querytarget(body)
    assert q["x"] == 1.5 and q["z"] == -3.0 and q["yaw"] == 45.0
    assert parse_querytarget({"statusCode": 0}) is None


def test_bedrock_backend_with_mock_client():
    pytest.importorskip("websockets")
    from mock_bedrock import MockBedrock

    port = free_port()
    be = BedrockWSBackend(host="127.0.0.1", port=port, wait_timeout=15, log=lambda *_: None,
                          probe_every=2)
    mock = MockBedrock(port).start()
    try:
        be.reset()
        assert "PlayerMessage" in mock.subscribed
        p = mock.world.player
        x0, z0, yaw0 = p.x, p.z, p.yaw
        for i in range(30):
            o = be.step(Action(forward=1.0, turn=0.15), 0.05)
        assert np.hypot(p.x - x0, p.z - z0) > 0.5 or be._touch > 0
        dyaw = ((p.yaw - yaw0 + 180) % 360) - 180
        assert dyaw > 0  # + = 左回り
        assert o.frame is not None and o.frame.shape[2] == 3
        assert "x" in o.info
        assert any("if block ~ ~ ~ air" in c for c in mock.commands)
    finally:
        mock.stop()
        be.close()


def test_dashboard_endpoints():
    from flycraft.backends.sim import SimWorld
    from flycraft.dashboard.server import serve
    from flycraft.fly import Fly, FlyConfig
    from flycraft.retinotopy import infer
    from flycraft.runner import Session
    from flycraft.toy import build_toy

    con = build_toy()
    fly = Fly(con, infer(con), FlyConfig(seed=0))
    s = Session(SimWorld(seed=0, width=64, height=36), fly, realtime=False)
    for _ in range(4):
        s.step_once()
    s._publish()
    port = free_port()
    httpd = serve(s, "127.0.0.1", port)
    try:
        base = f"http://127.0.0.1:{port}"
        html = urllib.request.urlopen(base + "/").read().decode()
        assert "FlyCraft" in html
        st = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert st["tick"] == 4 and "motor" in st
        br = json.loads(urllib.request.urlopen(base + "/api/brain").read())
        assert br["n"] == con.n
        mm = json.loads(urllib.request.urlopen(base + "/api/minimap").read())
        assert mm["w"] > 0
        req = urllib.request.Request(base + "/api/control", data=json.dumps({"type": "opto", "group": "p9",
                                                                             "value": True}).encode(), method="POST")
        urllib.request.urlopen(req).read()
        s._apply_controls()
        assert fly.opto["p9"]
    finally:
        httpd.shutdown()


def test_cli_probe_with_toy(capsys):
    from flycraft.cli import main

    main(["probe", "sugar", "--toy", "--ms", "300", "--seed", "0"])
    out = capsys.readouterr().out
    assert "feed_both" in out and "発火したニューロン" in out


def test_bedrock_turn_speed_and_telemetry():
    pytest.importorskip("websockets")
    from mock_bedrock import MockBedrock

    port = free_port()
    be = BedrockWSBackend(host="127.0.0.1", port=port, wait_timeout=15, log=lambda *_: None)
    mock = MockBedrock(port).start()
    try:
        be.reset()
        assert "gamerule sendcommandfeedback false" in mock.commands
        be.set_turn_speed(360)
        p = mock.world.player
        yaw0 = p.yaw
        t0 = time.time()
        while time.time() - t0 < 1.0:
            be.step(Action(turn=0.25), 0.05)
        time.sleep(0.2)
        dyaw = ((p.yaw - yaw0 + 180) % 360) - 180
        assert 65 < dyaw < 115  # 0.25 × 360°/秒 × 1 秒 = 90°（+ = 左）
        ex = be.extra_telemetry()
        assert ex["turn_deg_s"] == 360 and ex["turn_rate"] > 50 and ex["status"] == "操作中"
    finally:
        mock.stop()
        be.close()

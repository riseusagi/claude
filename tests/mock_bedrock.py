"""テスト用の「疑似 Minecraft 統合版クライアント」。

本物の統合版と同じ JSON メッセージ形式で FlyCraft の WebSocket サーバーに接続し、
FlyCraft が送るコマンド（tp / execute if block / querytarget など）を
内蔵ワールド (SimWorld) の上で解釈して応答する。
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
import time

from flycraft.backends.sim import CACTUS, EYE, FLOWER, JUMP_V, SimWorld
from flycraft.interface import Action

NAME_TO_BLOCK = {
    "dandelion": FLOWER, "poppy": FLOWER, "yellow_flower": FLOWER, "red_flower": FLOWER,
    "cactus": CACTUS, "air": 0,
}
FAIL = -2147483648

RE_MOVE = re.compile(r"execute as @s at @s rotated ~ 0 run tp @s \^ \^ \^(-?[\d.]+) ~(-?[\d.]+) (-?[\d.]+) true")
RE_JUMP = re.compile(r"execute as @s at @s unless block ~ ~-0.2 ~ air run tp @s ~ ~1.25 ~ true")
RE_PROBE = re.compile(r"execute as @s at @s anchored eyes rotated ~(-?[\d.]+) (-?[\d.]+) positioned \^ \^ \^([\d.]+) if block ~ ~ ~ (\w+)")
RE_TASTE = re.compile(r"execute as @s at @s if block ~ ~(-1)? ~ (\w+)$")


class MockBedrock:
    def __init__(self, port: int, seed: int = 0) -> None:
        self.port = port
        self.world = SimWorld(seed=seed, width=32, height=18)
        self.commands = []
        self.subscribed = []
        self.ws = None
        self._stop = False
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop = True

    # ------------------------------------------------------------ protocol
    def handle(self, cmd: str):
        w, p = self.world, self.world.player
        self.commands.append(cmd)
        if cmd == "querytarget @s":
            details = [{"dimension": 0, "position": {"x": p.x, "y": p.y + EYE, "z": p.z},
                        "uniqueId": "-4294967295", "yRot": ((-p.yaw + 180) % 360) - 180}]
            return {"statusCode": 0, "details": json.dumps(details), "statusMessage": "ok"}
        m = RE_MOVE.match(cmd)
        if m:
            dist, dyaw, pitch = float(m.group(1)), float(m.group(2)), float(m.group(3))
            p.yaw = (p.yaw - dyaw) % 360
            p.pitch = -pitch
            fx, fz = w._forward_vec()
            w._move(fx * dist, fz * dist)
            return {"statusCode": 0}
        if RE_JUMP.match(cmd):
            if p.on_ground:
                p.vy = JUMP_V
                p.on_ground = False
            return {"statusCode": 0}
        m = RE_PROBE.match(cmd)
        if m:
            ph, pitch, d, name = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)
            yaw = math.radians(p.yaw - ph)
            el = math.radians(-pitch)
            q = (p.x + math.sin(yaw) * math.cos(el) * d, p.y + EYE + math.sin(el) * d,
                 p.z + math.cos(yaw) * math.cos(el) * d)
            ok = w._block_at(*q) == NAME_TO_BLOCK.get(name, -1)
            return {"statusCode": 0 if ok else FAIL}
        m = RE_TASTE.match(cmd)
        if m:
            dy = -1 if m.group(1) else 0
            ok = w._block_at(p.x, p.y + dy + 0.1, p.z) == NAME_TO_BLOCK.get(m.group(2), -1)
            return {"statusCode": 0 if ok else FAIL}
        return {"statusCode": 0, "statusMessage": "ok"}

    async def _client(self):
        import websockets

        async with websockets.connect(f"ws://127.0.0.1:{self.port}") as ws:
            self.ws = ws

            async def physics():
                while not self._stop:
                    self.world.step(Action(), 0.05)
                    await asyncio.sleep(0.05)

            task = asyncio.ensure_future(physics())
            try:
                while not self._stop:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 0.2)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        return
                    msg = json.loads(raw)
                    h = msg["header"]
                    if h["messagePurpose"] == "subscribe":
                        self.subscribed.append(msg["body"]["eventName"])
                        continue
                    body = self.handle(msg["body"]["commandLine"])
                    await ws.send(json.dumps({"header": {"version": 1, "requestId": h["requestId"],
                                                         "messagePurpose": "commandResponse"},
                                              "body": body}))
            finally:
                task.cancel()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        for _ in range(50):
            try:
                self.loop.run_until_complete(self._client())
                return
            except OSError:
                time.sleep(0.1)

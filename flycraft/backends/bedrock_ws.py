"""Minecraft 統合版の WebSocket 接続（/connect）でハエに操作させる。

統合版はチャットで ``/connect <host>:<port>`` と入力すると外部の WebSocket
サーバーに接続し、コマンドの実行やイベントの購読ができる（MOD 不要。
Windows / iOS / Android など）。このモジュールがそのサーバーになる。

* 体の動き: ``tp`` の相対移動（前進・旋回）、段差ではジャンプ（上方向の tp）
* 噛む: 視線の先のブロックを ``setblock ... destroy`` で壊す／近くの生き物に ``damage``
* 感覚:
  - 視覚: 既定では ``execute ... if block ~ ~ ~ air`` の光線プローブで粗い奥行き画像を作る。
    同じ PC で遊んでいるなら ``--screen-vision`` で画面キャプチャを使う方がよく見える。
  - 足先の味覚: 足元の花・甘いベリー・蜂蜜ブロック → 甘味、サボテン・マグマ・火 → 苦味
  - 接触: 前進しようとしたのに進めなかった → 頭の剛毛
  - アイテムを拾った → 甘味、死んだ → 苦味
* チャットで ``!fly stop`` / ``!fly go`` / ``!fly hunger 0.8`` で操作できる。

設定: 統合版の「設定 > 一般 > 暗号化された WebSocket を要求」をオフにし、
チートが有効なワールドで ``/connect localhost:19131`` を実行する。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..interface import Action, Backend, Observation

EVENTS = ["PlayerMessage", "ItemAcquired", "PlayerDied", "BlockBroken", "MobKilled", "PlayerTravelled"]
SWEET_FEET = ["dandelion", "poppy", "yellow_flower", "red_flower", "cornflower", "azure_bluet",
              "oxeye_daisy", "sweet_berry_bush", "cake"]
SWEET_BELOW = ["honey_block"]
BITTER_FEET = ["fire", "lava", "wither_rose", "sweet_berry_bush"]
BITTER_BELOW = ["magma", "cactus"]
WALK_SPEED = 4.3  # ブロック/秒
TURN_SPEED = 360.0  # 旋回指令 1.0 のときの回転速度 [度/秒]（既定）


def _uuid() -> str:
    return str(uuid.uuid4())


def command_message(cmd: str, rid: Optional[str] = None) -> Tuple[str, str]:
    rid = rid or _uuid()
    msg = {
        "header": {"version": 1, "requestId": rid, "messageType": "commandRequest",
                   "messagePurpose": "commandRequest"},
        "body": {"version": 1, "commandLine": cmd, "origin": {"type": "player"}},
    }
    return rid, json.dumps(msg)


def subscribe_message(event: str) -> str:
    return json.dumps({
        "header": {"version": 1, "requestId": _uuid(), "messageType": "commandRequest",
                   "messagePurpose": "subscribe"},
        "body": {"eventName": event},
    })


def parse_querytarget(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """querytarget の応答から位置と向きを取り出す。"""
    raw = body.get("details")
    if raw is None:
        m = re.search(r"\[.*\]", body.get("statusMessage", ""), re.S)
        raw = m.group(0) if m else None
    if not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or "position" not in data:
        return None
    p = data["position"]
    return {"x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"]),
            "yaw": float(data.get("yRot", 0.0)), "dimension": data.get("dimension")}


# ------------------------------------------------------------------- bridge
class BedrockBridge:
    """WebSocket サーバー（別スレッドの asyncio ループ）。スレッドセーフな API を提供する。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 19131, max_inflight: int = 90,
                 log: Callable[[str], None] = print) -> None:
        self.host, self.port = host, port
        self.max_inflight = max_inflight
        self.log = log
        self.loop = asyncio.new_event_loop()
        self.ws = None
        self.connected = threading.Event()
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.pending: Dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bedrock-ws", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("WebSocket サーバーを起動できませんでした")
        if getattr(self, "_error", None):
            raise self._error

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            import websockets
        except ImportError:  # pragma: no cover
            self._error = RuntimeError("websockets が必要です: pip install websockets")
            self._ready.set()
            return

        async def main():
            try:
                self.server = await websockets.serve(self._handler, self.host, self.port,
                                                     max_size=None, ping_interval=None)
            except OSError as e:
                self._error = e
                self._ready.set()
                return
            self._ready.set()
            await asyncio.Future()

        try:
            self.loop.run_until_complete(main())
        except Exception:  # pragma: no cover
            pass

    async def _handler(self, ws, path=None):
        if self.ws is not None:
            self.log("🔁 新しい接続が来たので切り替えます")
        self.ws = ws
        self.connected.set()
        self.log("✅ Minecraft が接続しました")
        try:
            async for raw in ws:
                self._on_message(raw)
        except Exception:
            pass
        finally:
            if self.ws is ws:
                self.ws = None
                self.connected.clear()
                self.log("⚠ Minecraft との接続が切れました（もう一度 /connect してください）")
                with self._lock:
                    for f in self.pending.values():
                        if not f.done():
                            f.set_exception(ConnectionError("disconnected"))
                    self.pending.clear()

    def _on_message(self, raw) -> None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        header = msg.get("header", {})
        body = msg.get("body", {})
        purpose = header.get("messagePurpose")
        if purpose in ("commandResponse", "error"):
            rid = header.get("requestId")
            with self._lock:
                fut = self.pending.pop(rid, None)
            if fut is not None and not fut.done():
                fut.set_result(body)
        elif purpose == "event":
            name = header.get("eventName") or body.get("eventName")
            self.events.put({"name": name, "body": body})

    # ------------------------------------------------------------ public
    @property
    def inflight(self) -> int:
        with self._lock:
            return len(self.pending)

    def command(self, cmd: str, optional: bool = False) -> Optional[concurrent.futures.Future]:
        """コマンドを送る（非同期）。optional=True なら混雑時に送らない。"""
        ws = self.ws
        if ws is None:
            return None
        with self._lock:
            if optional and len(self.pending) >= self.max_inflight:
                return None
            rid, text = command_message(cmd)
            fut: concurrent.futures.Future = concurrent.futures.Future()
            self.pending[rid] = fut
        asyncio.run_coroutine_threadsafe(self._send(ws, text), self.loop)
        return fut

    def run(self, cmd: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        fut = self.command(cmd)
        if fut is None:
            return None
        try:
            return fut.result(timeout)
        except Exception:
            return None

    def subscribe(self, event: str) -> None:
        ws = self.ws
        if ws is not None:
            asyncio.run_coroutine_threadsafe(self._send(ws, subscribe_message(event)), self.loop)

    async def _send(self, ws, text: str) -> None:
        try:
            await ws.send(text)
        except Exception:
            pass

    def drop_stale(self, max_age: float = 5.0) -> None:
        """応答が返らない古い要求を捨てる（混雑で詰まらないように）。"""
        now = time.time()
        with self._lock:
            for rid, fut in list(self.pending.items()):
                t0 = getattr(fut, "_t0", None)
                if t0 is None:
                    fut._t0 = now  # type: ignore[attr-defined]
                elif now - t0 > max_age:
                    self.pending.pop(rid, None)
                    if not fut.done():
                        fut.cancel()

    def close(self) -> None:
        try:
            if hasattr(self, "server"):
                self.loop.call_soon_threadsafe(self.server.close)
        except Exception:
            pass


# ------------------------------------------------------------------ backend
@dataclass
class ProbeEye:
    """光線プローブの格子（φ: 右が正, θ: 上が正）。"""

    cols: int = 7
    rows: int = 4
    fov_h: float = 100.0
    fov_v: float = 60.0
    depths: Tuple[float, ...] = (1.5, 3.0, 6.0, 12.0)
    phi: np.ndarray = field(init=False)
    theta: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.phi = np.linspace(-self.fov_h / 2, self.fov_h / 2, self.cols)
        self.theta = np.linspace(self.fov_v / 2, -self.fov_v / 2, self.rows)


class BedrockWSBackend(Backend):
    name = "bedrock-ws"

    def __init__(self, host: str = "0.0.0.0", port: int = 19131, fov_v: float = 70.0,
                 screen_vision: bool = False, window: str = "Minecraft", pitch: float = 8.0,
                 probe: Optional[ProbeEye] = None, probe_every: int = 4, taste_every: int = 4,
                 bridge: Optional[BedrockBridge] = None, wait_timeout: Optional[float] = None,
                 log: Callable[[str], None] = print) -> None:
        self.log = log
        self.bridge = bridge or BedrockBridge(host, port, log=log)
        self.fov_v = fov_v
        self.pitch = pitch
        self.probe = probe or ProbeEye()
        self.probe_every = probe_every
        self.taste_every = taste_every
        self.wait_timeout = wait_timeout
        self.capture = None
        if screen_vision:
            from .screen import ScreenCapture

            self.capture = ScreenCapture(window)
        self.state: Dict[str, Any] = {}
        self._pos_prev = None
        self._depth = np.full((self.probe.rows, self.probe.cols), np.inf)
        self._probe_futs: List[Tuple[int, int, int, concurrent.futures.Future]] = []
        self._taste_futs: List[Tuple[str, concurrent.futures.Future]] = []
        self._qt_fut = None
        self._sugar = self._bitter = 0.0
        self._touch = 0.0
        self._blocked = 0.0
        self._bite_cd = 0.0
        self._tick = 0
        self._t_last = time.perf_counter()
        self._t_step = 0.0
        self._cmd_forward = 0.0
        self._last_dyaw = 0.0
        self.paused_by_chat = False
        self.turn_deg_s = TURN_SPEED
        self.turn_rate = 0.0  # 実際に回している速さ [度/秒]
        self.status = "接続待ち"
        self.control_hook: Optional[Callable[[Dict[str, Any]], None]] = None
        self.events: List[str] = []

    @property
    def realtime(self) -> bool:
        return True

    # --------------------------------------------------------------- setup
    def reset(self) -> Observation:
        port = self.bridge.port
        if not self.bridge.connected.is_set():
            self.log("")
            self.log("🟩 Minecraft 統合版で次の操作をしてください:")
            self.log("   1. 設定 > 一般 >「暗号化された WebSocket を要求」をオフ")
            self.log("   2. チートを有効にしたワールドに入る")
            self.log(f"   3. チャットで  /connect localhost:{port}  と入力（別の端末からは PC の IP アドレス）")
            self.log("")
        t0 = time.time()
        while not self.bridge.connected.wait(5.0):
            if self.wait_timeout and time.time() - t0 > self.wait_timeout:
                raise TimeoutError("Minecraft からの接続がありません")
            self.log(f"   …接続待ち（/connect localhost:{port}）")
        for ev in EVENTS:
            self.bridge.subscribe(ev)
        # コマンドの実行結果（「テレポートしました」やエラー）がチャットに流れ続けないようにする
        self.bridge.command("gamerule sendcommandfeedback false")
        self.bridge.command('tellraw @s {"rawtext":[{"text":"§e🪰 FlyCraft: ハエの脳が体を動かします。 Esc でチャットを閉じてください。 !fly stop / !fly go"}]}')
        self.log("   チャットは Esc で閉じてください（開いたままだと画面が見えず、ハエの目にもチャットが映ります）")
        self.events.append("接続しました")
        r = self.bridge.run("querytarget @s")
        if r:
            q = parse_querytarget(r)
            if q:
                self.state.update(q)
        return self._observe(0.05)

    # ---------------------------------------------------------------- loop
    def step(self, action: Action, dt: float) -> Observation:
        self._tick += 1
        self._drain_events()
        self._collect()
        if not self.bridge.connected.is_set():
            time.sleep(dt)
            return self._observe(dt)
        now = time.perf_counter()
        wall_dt = min(0.2, max(0.0, now - self._t_step)) if self._t_step else 0.05
        self._t_step = now
        if not self.paused_by_chat:
            self._act(action, wall_dt if wall_dt > 0 else dt)
            self.status = "操作中"
        else:
            self.status = "チャットの !fly stop で停止中（!fly go で再開）"
        if wall_dt > 0:
            self.turn_rate += min(1.0, wall_dt) * (abs(self._last_dyaw) / wall_dt - self.turn_rate)
        self._query()
        # 実時間に合わせる
        now = time.perf_counter()
        wait = dt - (now - self._t_last)
        if wait > 0:
            time.sleep(wait)
        self._t_last = time.perf_counter()
        return self._observe(dt)

    def _act(self, action: Action, dt: float) -> None:
        b = self.bridge
        fwd = float(np.clip(action.forward, -1, 1))
        if fwd < 0:
            fwd *= 0.5
        dist = WALK_SPEED * fwd * dt
        dyaw = -self.turn_deg_s * float(np.clip(action.turn, -1, 1)) * dt  # MC の yaw は右回りが正
        dyaw = float(np.clip(dyaw, -60.0, 60.0))
        self._last_dyaw = dyaw
        self._cmd_forward = dist
        if abs(dist) > 1e-3 or abs(dyaw) > 0.05:
            b.command(f"execute as @s at @s rotated ~ 0 run tp @s ^ ^ ^{dist:.3f} ~{dyaw:.2f} {self.pitch:.1f} true")
        if action.jump:
            b.command("execute as @s at @s unless block ~ ~-0.2 ~ air run tp @s ~ ~1.25 ~ true")
        self._bite_cd -= dt
        if action.attack and self._bite_cd <= 0:
            self._bite_cd = 0.6
            b.command("execute as @s at @s anchored eyes positioned ^ ^ ^1.5 unless block ~ ~ ~ bedrock "
                      "unless block ~ ~ ~ air run setblock ~ ~ ~ air destroy")
            b.command("execute as @s at @s anchored eyes positioned ^ ^ ^1.5 run "
                      "damage @e[r=1.5,c=1,type=!player,type=!item] 2 entity_attack entity @s")

    def _query(self) -> None:
        b = self.bridge
        b.drop_stale()
        if self._qt_fut is None:
            self._qt_fut = b.command("querytarget @s", optional=True)
        if self._tick % self.taste_every == 0 and not self._taste_futs:
            for name in SWEET_FEET:
                self._taste_futs.append(("sweet", b.command(f"execute as @s at @s if block ~ ~ ~ {name}", optional=True)))
            for name in SWEET_BELOW:
                self._taste_futs.append(("sweet", b.command(f"execute as @s at @s if block ~ ~-1 ~ {name}", optional=True)))
            for name in BITTER_FEET:
                self._taste_futs.append(("bitter", b.command(f"execute as @s at @s if block ~ ~ ~ {name}", optional=True)))
            for name in BITTER_BELOW:
                self._taste_futs.append(("bitter", b.command(f"execute as @s at @s if block ~ ~-1 ~ {name}", optional=True)))
            self._taste_futs = [(k, f) for k, f in self._taste_futs if f is not None]
        if self.capture is None and self._tick % self.probe_every == 0 and not self._probe_futs:
            pe = self.probe
            for r, th in enumerate(pe.theta):
                for c, ph in enumerate(pe.phi):
                    pitch = self.pitch - th  # MC のピッチは下向きが正
                    for k, d in enumerate(pe.depths):
                        f = b.command(f"execute as @s at @s anchored eyes rotated ~{ph:.1f} {pitch:.1f} "
                                      f"positioned ^ ^ ^{d:.2f} if block ~ ~ ~ air", optional=True)
                        if f is not None:
                            self._probe_futs.append((r, c, k, f))

    def _collect(self) -> None:
        """届いた応答を反映する（待たない）。"""
        f = self._qt_fut
        if f is not None and f.done():
            self._qt_fut = None
            try:
                q = parse_querytarget(f.result())
            except Exception:
                q = None
            if q:
                prev = self.state.get("x"), self.state.get("z")
                self.state.update(q)
                if prev[0] is not None:
                    moved = math.hypot(q["x"] - prev[0], q["z"] - prev[1])
                    expect = abs(self._cmd_forward)
                    if expect > 0.05 and moved < 0.3 * expect:
                        self._blocked += 1
                    else:
                        self._blocked = 0
                    if self._blocked >= 3:
                        self._touch = 0.8
        if self._taste_futs and all(f.done() for _, f in self._taste_futs):
            sweet = bitter = False
            for kind, f in self._taste_futs:
                try:
                    ok = f.result().get("statusCode", -1) == 0
                except Exception:
                    ok = False
                if ok and kind == "sweet":
                    sweet = True
                if ok and kind == "bitter":
                    bitter = True
            if sweet:
                self._sugar = 1.0
            if bitter:
                self._bitter = 1.0
            self._taste_futs = []
        if self._probe_futs and all(f.done() for *_, f in self._probe_futs):
            depth = np.full((self.probe.rows, self.probe.cols), np.inf)
            for r, c, k, f in self._probe_futs:
                try:
                    air = f.result().get("statusCode", -1) == 0
                except Exception:
                    air = True
                if not air:
                    depth[r, c] = min(depth[r, c], self.probe.depths[k])
            self._depth = depth
            self._probe_futs = []

    def _drain_events(self) -> None:
        while True:
            try:
                ev = self.bridge.events.get_nowait()
            except queue.Empty:
                return
            name, body = ev.get("name"), ev.get("body", {})
            if name == "ItemAcquired":
                self._sugar = max(self._sugar, 0.8)
                self.events.append("アイテムを拾った（甘い）")
            elif name == "PlayerDied":
                self._bitter = 1.0
                self.events.append("死んでしまった（苦い）")
            elif name == "PlayerMessage":
                text = str(body.get("message", "")).strip()
                if text.startswith("!fly"):
                    self._chat_command(text[4:].strip())
            self.events = self.events[-20:]

    def _chat_command(self, arg: str) -> None:
        if arg in ("stop", "pause", "止まれ"):
            self.paused_by_chat = True
            self.events.append("チャット: 停止")
        elif arg in ("go", "start", "resume", "動け"):
            self.paused_by_chat = False
            self.events.append("チャット: 再開")
        elif arg.startswith("hunger") and self.control_hook:
            try:
                self.control_hook({"type": "hunger", "value": float(arg.split()[1])})
            except (IndexError, ValueError):
                pass

    # --------------------------------------------------------- observation
    def probe_frame(self, width: int = 140, height: int = 84) -> np.ndarray:
        """プローブの奥行きを粗い画像にする（近いほど暗い、何も無ければ空/遠景）。"""
        pe = self.probe
        img = np.zeros((pe.rows, pe.cols, 3), dtype=np.float32)
        far = max(pe.depths) * 1.5
        for r, th in enumerate(pe.theta):
            for c in range(pe.cols):
                d = self._depth[r, c]
                if np.isinf(d):
                    img[r, c] = (150, 190, 255) if th > -5 else (110, 140, 90)
                else:
                    v = 40 + 170 * min(1.0, d / far)
                    img[r, c] = (v * 0.8, v, v * 0.7)
        ys = np.minimum((np.arange(height) * pe.rows) // height, pe.rows - 1)
        xs = np.minimum((np.arange(width) * pe.cols) // width, pe.cols - 1)
        return img[ys][:, xs].astype(np.uint8)

    def _observe(self, dt: float) -> Observation:
        decay = math.exp(-dt / 0.4)
        self._sugar *= decay
        self._bitter *= decay
        self._touch *= math.exp(-dt / 0.3)
        if self.capture is not None:
            try:
                frame = self.capture.grab()
                fov = self.fov_v
            except Exception:
                frame, fov = self.probe_frame(), self.probe.fov_v
        else:
            frame, fov = self.probe_frame(), self.probe.fov_v
        info = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.state.items()}
        info["connected"] = self.bridge.connected.is_set()
        info["inflight"] = self.bridge.inflight
        return Observation(frame=frame, fov_v=fov, sugar=self._sugar, bitter=self._bitter,
                           touch_left=self._touch, touch_right=self._touch, info=info)

    def set_turn_speed(self, deg_s: float) -> None:
        self.turn_deg_s = float(np.clip(deg_s, 30, 1440))

    def extra_telemetry(self) -> Dict[str, Any]:
        return {"events": list(self.events[-8:]), "status": self.status,
                "turn_deg_s": round(self.turn_deg_s), "turn_rate": round(self.turn_rate)}

    def close(self) -> None:
        self.bridge.close()

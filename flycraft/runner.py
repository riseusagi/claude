"""ゲーム ⇄ ハエ の実行ループ（別スレッド）と、ダッシュボード向けの状態共有。"""

from __future__ import annotations

import base64
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from .fly import OPTO_GROUPS, Fly
from .interface import Action, Backend, Observation


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


class Session:
    """backend と fly を dt ごとに交互に進める。

    * シミュレータ（非実時間）: ゲーム時間 = 脳の時間。realtime=True なら実時間に合わせて待つ。
    * Minecraft 本体（実時間）: 毎ループ dt 分だけ脳を進める。脳が遅いとハエの反応が遅れる。
    """

    def __init__(self, backend: Backend, fly: Fly, dt_ms: float = 50.0, realtime: bool = True,
                 log: Callable[[str], None] = print) -> None:
        self.backend = backend
        self.fly = fly
        self.dt_ms = dt_ms
        self.realtime = realtime
        self.log = log
        self.paused = False
        self.running = False
        self.controls: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._snapshot: Dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None
        self.obs: Optional[Observation] = None
        self.action = Action()
        self.loop_hz = 0.0
        self.tick = 0
        self.error: Optional[str] = None
        self._t_prev: Optional[float] = None
        self.last_dt_ms = dt_ms

    # ------------------------------------------------------------- control
    def start(self) -> None:
        self.running = True
        self._thread = threading.Thread(target=self._run, name="flycraft-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        try:
            self.backend.close()
        except Exception:
            pass

    def control(self, msg: Dict[str, Any]) -> None:
        self.controls.put(msg)

    def _apply_controls(self) -> None:
        while True:
            try:
                msg = self.controls.get_nowait()
            except queue.Empty:
                return
            kind = msg.get("type")
            if kind == "pause":
                self.paused = bool(msg.get("value", not self.paused))
            elif kind == "opto" and msg.get("group") in OPTO_GROUPS:
                self.fly.opto[msg["group"]] = bool(msg.get("value"))
            elif kind == "hunger":
                self.fly.hunger = float(np.clip(msg.get("value", 0.5), 0, 1))
            elif kind == "reset_brain":
                self.fly.brain.reset()
            elif kind == "reset_world" and hasattr(self.backend, "reset"):
                self.obs = self.backend.reset()

    # ---------------------------------------------------------------- loop
    def step_once(self) -> Action:
        """1 ステップ（テストやヘッドレス実行用）。"""
        if self.obs is None:
            self.obs = self.backend.reset()
        dt_ms = self.dt_ms
        now = time.perf_counter()
        if self.backend.realtime and self._t_prev is not None:
            # 実時間で動くゲームでは、前回からの経過時間ぶん脳を進める（脳の時間 = 実時間）。
            # 脳が遅いと 1 回の刻みが大きくなるが、上限（2 倍）で止めて遅れを溜めない。
            dt_ms = float(np.clip((now - self._t_prev) * 1000.0, self.dt_ms, 2 * self.dt_ms))
        self._t_prev = now
        self.last_dt_ms = dt_ms
        self.action = self.fly.step(self.obs, dt_ms)
        self.obs = self.backend.step(self.action, self.dt_ms / 1000.0)
        self.tick += 1
        return self.action

    def _run(self) -> None:
        with self._lock:
            self._snapshot = {"tick": 0, "backend": self.backend.name, "status": "ゲームへの接続を待っています…"}
        try:
            self.obs = self.backend.reset()
        except Exception as e:  # pragma: no cover - 接続エラーなど
            self.error = f"{type(e).__name__}: {e}"
            self.log(f"[flycraft] ゲームに接続できません: {self.error}")
            self.running = False
            return
        last = time.perf_counter()
        ema = None
        while self.running:
            t0 = time.perf_counter()
            self._apply_controls()
            if self.paused:
                self._publish()
                time.sleep(0.05)
                last = time.perf_counter()
                continue
            try:
                self.step_once()
            except Exception as e:  # pragma: no cover - 表示して継続
                import traceback

                self.error = f"{type(e).__name__}: {e}"
                self.log("[flycraft] ループでエラー:\n" + traceback.format_exc())
                time.sleep(0.5)
                continue
            if self.tick % 2 == 0:
                self._publish()
            now = time.perf_counter()
            inst = 1.0 / max(1e-6, now - last)
            ema = inst if ema is None else 0.9 * ema + 0.1 * inst
            self.loop_hz = ema
            last = now
            if self.realtime and not self.backend.realtime:
                wait = self.dt_ms / 1000.0 - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)

    # ----------------------------------------------------------- telemetry
    def _publish(self) -> None:
        fly = self.fly
        snap: Dict[str, Any] = {"tick": self.tick, "paused": self.paused, "loop_hz": round(self.loop_hz, 1),
                                "backend": self.backend.name, "error": self.error, "status": ""}
        snap.update(fly.telemetry())
        obs = self.obs
        if obs is not None:
            snap["info"] = obs.info
            if obs.frame is not None:
                f = obs.frame
                step = max(1, int(np.ceil(f.shape[1] / 240)))
                small = f[::step, ::step, :3]
                snap["frame"] = {"w": small.shape[1], "h": small.shape[0], "rgb": _b64(small)}
        v = fly.view
        if v:
            rgb = v["rgb"]
            snap["eye"] = {"w": rgb.shape[1], "h": rgb.shape[0], "rgb": _b64(rgb)}
            feats = v["feats"]
            mot = np.sqrt(feats["hx"] ** 2 + feats["vy"] ** 2)
            m = float(np.percentile(mot, 99)) + 1e-6
            snap["motion"] = _b64(np.clip(mot / m * 255, 0, 255).astype(np.uint8))
        act = fly.activity
        top = np.flatnonzero(act > 0.05)
        if len(top) > 40000:
            top = top[np.argsort(-act[top])[:40000]]
        snap["act_idx"] = _b64(top.astype(np.int32))
        snap["act_val"] = _b64(np.clip(act[top] * 60, 0, 255).astype(np.uint8))
        try:
            snap.update(self.backend.extra_telemetry())
        except Exception:
            pass
        with self._lock:
            self._snapshot = snap

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

"""ハエ（脳 + 感覚器 + 運動出力）をひとまとめにしたもの。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .brain import Brain, LIFParams
from .connectome import Connectome
from .interface import Action, Observation
from .motor import LABELS as MOTOR_LABELS
from .motor import MotorDecoder, MotorParams, ObstacleReflex, ReflexParams
from .retinotopy import Retinotopy, photoreceptor_directions
from .senses import Senses
from .vision import Camera, CompoundEye, OpticLobe, VisualProjection


@dataclass
class FlyConfig:
    acuity: float = 3.0  # 複眼格子の間隔 [度]
    hunger: float = 0.6  # 初期の空腹度 (0〜1)
    hunger_rate: float = 0.004  # 1 秒あたりの空腹度の増加
    satiation: float = 0.25  # 甘味 1 秒あたりの満腹化
    walk_bias_base: float = 5.0  # 空腹度 0 のときの前進指令ニューロンへの電流 [mV]
    walk_bias_span: float = 6.5  # 空腹度 1 で追加される電流 [mV]（7 mV 超で自発発火）
    retina: bool = False  # 視細胞も直接駆動する（行動への影響はほぼ無いが脳活動の表示が増える）
    seed: Optional[int] = None
    engine: str = "auto"
    threads: Optional[int] = None
    lif: LIFParams = field(default_factory=LIFParams)
    motor: MotorParams = field(default_factory=MotorParams)
    reflex: ReflexParams = field(default_factory=ReflexParams)


# 光遺伝学パネル（ダッシュボード）で操作できるニューロン群
OPTO_GROUPS: Dict[str, Dict[str, Any]] = {
    "sugar": {"label": "甘味受容ニューロン", "select": {"cell_sub_class": "sugar/water"}, "mode": "rate", "value": 150.0},
    "bitter": {"label": "苦味受容ニューロン", "select": {"cell_sub_class": "bitter"}, "mode": "rate", "value": 150.0},
    "lplc2": {"label": "LPLC2（ルーミング検出）", "select": {"cell_type": "LPLC2"}, "mode": "rate", "value": 120.0},
    "gf": {"label": "巨大繊維 DNp01", "select": {"cell_type": "DNp01"}, "mode": "rate", "value": 80.0},
    "p9": {"label": "P9 DNp09（前進）", "select": {"cell_type": "DNp09"}, "mode": "bias", "value": 12.0},
    "mdn": {"label": "MDN（後退）", "select": {"cell_type": "MDN"}, "mode": "bias", "value": 12.0},
    "dna02_l": {"label": "DNa02 左（左旋回）", "select": {"cell_type": "DNa02", "side": "left"}, "mode": "bias", "value": 12.0},
    "dna02_r": {"label": "DNa02 右（右旋回）", "select": {"cell_type": "DNa02", "side": "right"}, "mode": "bias", "value": 12.0},
    "silence_gf": {"label": "巨大繊維を抑制", "select": {"cell_type": "DNp01"}, "mode": "bias", "value": -30.0},
}


class Fly:
    def __init__(self, con: Connectome, retino: Retinotopy, config: Optional[FlyConfig] = None) -> None:
        self.con = con
        self.retino = retino
        self.cfg = config or FlyConfig()
        self.brain = Brain(con, self.cfg.lif, engine=self.cfg.engine, seed=self.cfg.seed,
                           threads=self.cfg.threads)
        self.senses = Senses(con)
        self.motor = MotorDecoder(con, self.cfg.motor)
        self.reflex = ObstacleReflex(self.cfg.reflex, seed=self.cfg.seed)
        self.hunger = float(self.cfg.hunger)
        self.walk_idx = con.select(cell_type="DNp09")
        self._vision_key = None
        self.eye: Optional[CompoundEye] = None
        self.lobe: Optional[OpticLobe] = None
        self.vpn: Optional[VisualProjection] = None
        self._pr = None
        self.opto: Dict[str, bool] = {k: False for k in OPTO_GROUPS}
        self._opto_idx = {k: con.select(**g["select"]) for k, g in OPTO_GROUPS.items()}
        self.activity = np.zeros(con.n, dtype=np.float32)
        self.sc_names, self.sc_codes = np.unique(con.ann["super_class"], return_inverse=True)
        self.last_counts = np.zeros(con.n, dtype=np.int32)
        self.stats = {"steps": 0, "brain_ms": 0.0, "wall_brain": 0.0, "jumps": 0, "bites": 0}
        self.view: Dict[str, Any] = {}

    # ----------------------------------------------------------------- vision
    def _ensure_vision(self, frame: np.ndarray, fov_v: float) -> None:
        cam = Camera(frame.shape[1], frame.shape[0], fov_v)
        key = cam.key()
        if key == self._vision_key:
            return
        self.eye = CompoundEye(cam, self.cfg.acuity)
        self.lobe = OpticLobe(self.eye.shape, self.cfg.acuity)
        self.vpn = VisualProjection(self.con, self.retino, self.eye)
        self._vision_key = key
        # 視細胞 → 最寄りの格子点（視野内のもののみ）
        idx, phi, theta = photoreceptor_directions(self.con)
        fh, fv = cam.fov_h, cam.fov_v
        inside = (np.abs(phi) < fh / 2) & (np.abs(theta) < fv / 2)
        col = np.clip(((phi + fh / 2) / fh * self.eye.cols).astype(int), 0, self.eye.cols - 1)
        row = np.clip(((fv / 2 - theta) / fv * self.eye.rows).astype(int), 0, self.eye.rows - 1)
        self._pr = (idx[inside], row[inside], col[inside])

    # ------------------------------------------------------------------- step
    def step(self, obs: Observation, dt_ms: float) -> Action:
        dt = dt_ms / 1000.0
        n = self.con.n
        rates = np.zeros(n, dtype=np.float32)
        upper = 0.5
        if obs.frame is not None:
            self._ensure_vision(obs.frame, obs.fov_v)
            L, rgb = self.eye.sample(obs.frame)
            feats = self.lobe.update(L, dt)
            np.maximum(rates, self.vpn.rates(feats, dt, n), out=rates)
            upper = float(L[: max(1, L.shape[0] // 3)].mean())
            if self.cfg.retina and self._pr is not None:
                idx, r, c = self._pr
                rates[idx] = np.maximum(rates[idx], 5.0 + 60.0 * L[r, c])
            self.view = {"lum": L, "rgb": rgb, "feats": feats}
        self.senses.rates(obs, upper, dt, rates)

        # 空腹 → 前進指令ニューロン (P9) への定常電流。甘味で満たされる。
        self.hunger = float(np.clip(self.hunger + self.cfg.hunger_rate * dt
                                    - self.cfg.satiation * obs.sugar * dt, 0.0, 1.0))
        self.brain.clear_bias()
        self.brain.set_bias(self.walk_idx, self.cfg.walk_bias_base + self.cfg.walk_bias_span * self.hunger)

        for key, on in self.opto.items():
            if not on:
                continue
            g = OPTO_GROUPS[key]
            idx = self._opto_idx[key]
            if g["mode"] == "rate":
                rates[idx] = np.maximum(rates[idx], g["value"])
            else:
                self.brain.bias[idx] += g["value"]

        self.brain.set_drive_dense(rates)
        t0 = time.perf_counter()
        counts = self.brain.run(dt_ms)
        self.stats["wall_brain"] += time.perf_counter() - t0
        self.stats["brain_ms"] += dt_ms
        self.stats["steps"] += 1
        action = self.motor.decode(counts, dt_ms)
        action = self.reflex.update(obs, action, dt)
        self.action = action
        self.stats["jumps"] += int(action.jump)
        self.stats["bites"] += int(action.attack)
        self.last_counts = counts
        decay = np.float32(np.exp(-dt / 0.25))
        self.activity *= decay
        self.activity += counts
        return action

    # -------------------------------------------------------------- telemetry
    def realtime_factor(self) -> float:
        w = self.stats["wall_brain"]
        return (self.stats["brain_ms"] / 1000.0) / w if w > 0 else 0.0

    def class_activity(self) -> Dict[str, int]:
        active = self.last_counts > 0
        cnt = np.bincount(self.sc_codes[active], minlength=len(self.sc_names))
        return {str(k or "unknown"): int(v) for k, v in zip(self.sc_names, cnt)}

    def telemetry(self) -> Dict[str, Any]:
        m = self.motor
        return {
            "hunger": self.hunger,
            "action": getattr(self, "action", m.last).to_dict(),
            "brain_action": m.last.to_dict(),
            "reflex": {"active": self.reflex.active, "count": self.reflex.count, "enabled": self.reflex.p.enabled},
            "motor": {k: round(v, 2) for k, v in m.rates.items()},
            "motor_labels": MOTOR_LABELS,
            "senses": {k: round(v, 3) for k, v in self.senses.values.items()},
            "sense_labels": {k: g.name for k, g in self.senses.groups.items()},
            "spikes": int(self.last_counts.sum()),
            "active": int((self.last_counts > 0).sum()),
            "classes": self.class_activity(),
            "realtime": round(self.realtime_factor(), 2),
            "brain_t": round(self.brain.t_ms / 1000.0, 2),
            "opto": dict(self.opto),
            "vpn_driven": self.vpn.n_driven if self.vpn else 0,
            "vpn": {k: round(float(v.mean()), 2) for k, v in (self.vpn.last.items() if self.vpn else [])},
            "gf_spikes": getattr(m, "gf_spikes", 0),
            "stats": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.stats.items()},
        }

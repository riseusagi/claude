"""下行性ニューロン (DN) / 運動ニューロンの活動 → ゲーム操作。

ハエの脳は、約 1,300 本の下行性ニューロンで胸部神経節（歩行・飛行の
パターン生成器）に指令を送る。ここでは行動との対応がよく調べられている
ものだけを読み出す:

==========  =======================  ==========================================
行動         ニューロン                根拠
==========  =======================  ==========================================
前進         DNp09 (P9), DNg100 (BDN2)  Bidaye et al. 2020
後退         MDN (moonwalker)           Bidaye et al. 2014
旋回         DNa02, DNa01（同側へ旋回）  Rayshubskiy et al. 2020 ほか
逃避ジャンプ  DNp01（巨大繊維）, DNp02/04/11  von Reyn et al. 2014, Ache et al. 2019
噛む（摂食）  MN9（口吻伸展運動ニューロン）  Shiu et al. 2024
==========  =======================  ==========================================

胸部神経節そのもの（歩脚の協調など）は FlyWire の脳データに含まれないので、
「体」側で単純化して扱う: 前進・旋回の指令を移動速度と向きの変化に変換する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .connectome import Connectome
from .interface import Action, Observation

# Shiu et al. 2024 が口吻伸展の読み出しに用いた MN9 の root ID（FlyWire v783）
MN9_ROOT_IDS = (720575940660219265, 720575940618238523)

MOTOR_MAP: Dict[str, List[Tuple[str, float]]] = {
    "forward": [("DNp09", 1.0), ("DNg100", 0.6)],
    "backward": [("MDN", 1.0)],
    "steer": [("DNa02", 1.0), ("DNa01", 0.6)],
    "escape": [("DNp01", 1.0), ("DNp02", 0.3), ("DNp04", 0.3), ("DNp11", 0.3)],
}

LABELS = {
    "forward": "前進 (P9/BDN2)",
    "backward": "後退 (MDN)",
    "steer": "旋回 (DNa01/02)",
    "escape": "逃避 (巨大繊維)",
    "feed": "摂食 (MN9)",
}


@dataclass
class MotorParams:
    tau: float = 0.15  # 発火率の平滑化 [s]
    walk_full_hz: float = 25.0  # この発火率で全速前進
    turn_full_hz: float = 15.0  # 左右差がこの値で最大旋回
    backward_gain: float = 1.5
    feed_hz: float = 4.0  # MN9 がこれを超えたら噛む
    escape_hz: float = 60.0  # 逃避系の重み付き発火率がこれを超えたら跳ぶ


@dataclass
class _Pop:
    idx: np.ndarray
    w: np.ndarray


class MotorDecoder:
    def __init__(self, con: Connectome, params: MotorParams = None) -> None:
        self.p = params or MotorParams()
        self.pops: Dict[Tuple[str, str], _Pop] = {}
        for name, items in MOTOR_MAP.items():
            for side in ("left", "right"):
                idx, w = [], []
                for ct, weight in items:
                    sel = con.select(cell_type=ct, side=side)
                    idx.extend(sel.tolist())
                    w.extend([weight] * len(sel))
                self.pops[(name, side)] = _Pop(np.array(idx, dtype=np.int64), np.array(w, np.float32))
        mn9 = con.index_of_root(MN9_ROOT_IDS)
        if len(mn9) == 0:
            mn9 = con.select(cell_type="MN9")
        self.pops[("feed", "both")] = _Pop(mn9, np.ones(len(mn9), np.float32))
        self.gf = np.concatenate(
            [con.select(cell_type="DNp01", side=s) for s in ("left", "right")]
        ).astype(np.int64)
        self.rates: Dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        self.rates = {f"{k[0]}_{k[1]}": 0.0 for k in self.pops}
        self.last = Action()

    def neurons(self) -> Dict[str, np.ndarray]:
        return {f"{k[0]}_{k[1]}": v.idx for k, v in self.pops.items()}

    def decode(self, counts: np.ndarray, dt_ms: float) -> Action:
        p = self.p
        dt = dt_ms / 1000.0
        a = min(1.0, dt / p.tau)
        for (name, side), pop in self.pops.items():
            key = f"{name}_{side}"
            if len(pop.idx):
                inst = float((counts[pop.idx] * pop.w).sum() / max(pop.w.sum(), 1e-6)) / dt
            else:
                inst = 0.0
            self.rates[key] += a * (inst - self.rates[key])
        r = self.rates
        fwd = 0.5 * (r["forward_left"] + r["forward_right"])
        bwd = 0.5 * (r["backward_left"] + r["backward_right"])
        drive = (fwd - p.backward_gain * bwd) / p.walk_full_hz
        forward = float(np.clip(drive, -1.0, 1.0))
        turn = float(np.clip((r["steer_left"] - r["steer_right"]) / p.turn_full_hz, -1.0, 1.0))
        gf_spikes = int(counts[self.gf].sum()) if len(self.gf) else 0
        esc = 0.5 * (r["escape_left"] + r["escape_right"])
        jump = gf_spikes > 0 or esc > p.escape_hz
        attack = r["feed_both"] > p.feed_hz
        self.last = Action(forward=forward, turn=turn, jump=bool(jump), attack=bool(attack))
        self.gf_spikes = gf_spikes
        return self.last


@dataclass
class ReflexParams:
    enabled: bool = True
    stuck_time: float = 0.5  # 障害物を押し続けてから反射が起きるまで [s]
    turn_time: float = 0.8  # 向きを変える時間 [s]


class ObstacleReflex:
    """胸部神経節（VNC）の障害物回避反射の簡略モデル。

    ハエは脚や触角が壁に当たると、脳からの指令とは別に胸部の回路で歩行を
    調整する。FlyWire の「脳」データには胸部神経節が含まれないため、ここだけは
    コネクトームではなく単純な規則で実装している（--no-reflex で無効化できる）。
    接触情報そのものは剛毛の感覚ニューロンとして脳にも届いている。
    """

    def __init__(self, params: ReflexParams = None, seed=None) -> None:
        self.p = params or ReflexParams()
        self.rng = np.random.default_rng(seed)
        self.push = 0.0
        self.timer = 0.0
        self.direction = 0.0
        self.count = 0

    @property
    def active(self) -> bool:
        return self.timer > 0

    def update(self, obs: Observation, action: Action, dt: float) -> Action:
        if not self.p.enabled:
            return action
        tl, tr = obs.touch_left, obs.touch_right
        if self.timer > 0:
            self.timer -= dt
            return Action(forward=-0.3, turn=self.direction, jump=action.jump, attack=action.attack)
        if max(tl, tr) > 0.3 and action.forward > 0.2:
            self.push += dt
        else:
            self.push = max(0.0, self.push - dt)
        if self.push > self.p.stuck_time:
            self.push = 0.0
            self.timer = self.p.turn_time
            self.count += 1
            if abs(tl - tr) > 0.1:
                self.direction = -1.0 if tl > tr else 1.0  # 触れた側と反対へ（+ = 左）
            else:
                self.direction = float(self.rng.choice([-1.0, 1.0]))
        return action

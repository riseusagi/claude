"""ゲームとハエの間でやり取りするデータ型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Observation:
    """ゲームから体（感覚器）に届く情報。値はすべて 0〜1 程度に正規化する。"""

    frame: Optional[np.ndarray] = None  # (H, W, 3) uint8 RGB。一人称視点
    fov_v: float = 70.0  # frame の垂直視野角 [度]
    sugar: float = 0.0  # 甘味（花・食べ物に触れた）
    bitter: float = 0.0  # 苦味・痛み（サボテン・溶岩・ダメージ）
    touch_left: float = 0.0  # 頭部の剛毛（左）に何かが触れた（衝突）
    touch_right: float = 0.0
    wind: float = 0.0  # 触角で感じる風（移動速度・落下）
    odor_left: float = 0.0  # 左右の触角の匂い（食べ物の匂い）
    odor_right: float = 0.0
    sound: float = 0.0  # 触角の聴覚（大きな音）
    info: Dict[str, Any] = field(default_factory=dict)  # 表示用（座標・体力など）


@dataclass
class Action:
    """脳（下行性ニューロン）から体を経てゲームに送る操作。"""

    forward: float = 0.0  # -1（後退）〜 1（前進）
    turn: float = 0.0  # -1（右）〜 1（左）: 反時計回りが正
    jump: bool = False  # 逃避ジャンプ（巨大繊維）
    attack: bool = False  # 口吻伸展 → 噛む（ブロックを壊す・攻撃）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Backend:
    """ゲームとの接続口。"""

    name = "backend"

    def reset(self) -> Observation:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, action: Action, dt: float) -> Observation:  # pragma: no cover - interface
        """action を dt 秒間適用し、その後の観測を返す。"""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def extra_telemetry(self) -> Dict[str, Any]:
        return {}

    @property
    def realtime(self) -> bool:
        """True ならゲーム側の時間が実時間で流れる（Minecraft 本体）。"""
        return False

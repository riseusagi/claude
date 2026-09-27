"""視覚以外の感覚器 → 感覚ニューロンの発火率。

どのニューロンを使うかは FlyWire の注釈（Schlegel et al. 2024）で決める:

* 甘味 / 苦味: 唇弁の味覚受容ニューロン (cell_sub_class = sugar/water, bitter)
* 接触: 頭部と複眼の機械感覚剛毛 (head bristle, eye bristle)。左右別
* 風: ジョンストン器官の風・重力感受性ニューロン (wind_gravity)
* 聴覚: ジョンストン器官の聴覚ニューロン (auditory)
* 匂い: 食べ物の匂い（酢・果実）に応答する嗅覚受容ニューロン (ORN_DM1 など)。左右別
* 単眼: ocellar（画面上部の明暗変化）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .connectome import Connectome
from .interface import Observation

# 誘引性の匂い（酢酸エチル・果実臭など）に強く応答する糸球体
FOOD_ORNS = ["ORN_DM1", "ORN_DM4", "ORN_DM2", "ORN_VM2", "ORN_DP1m"]


@dataclass
class SenseGroup:
    name: str
    neurons: np.ndarray
    max_rate: float


class Senses:
    def __init__(self, con: Connectome) -> None:
        sel = con.select
        g = {}
        g["sugar"] = SenseGroup("甘味", sel(cell_sub_class="sugar/water"), 150.0)
        g["bitter"] = SenseGroup("苦味", sel(cell_sub_class="bitter"), 150.0)
        for s, key in (("left", "touch_left"), ("right", "touch_right")):
            g[key] = SenseGroup(
                f"接触({'左' if s == 'left' else '右'})",
                sel(cell_sub_class=["head bristle", "eye bristle"], side=s),
                120.0,
            )
        g["wind"] = SenseGroup("風", sel(cell_sub_class="wind_gravity"), 80.0)
        g["sound"] = SenseGroup("音", sel(cell_sub_class="auditory"), 100.0)
        for s, key in (("left", "odor_left"), ("right", "odor_right")):
            g[key] = SenseGroup(
                f"匂い({'左' if s == 'left' else '右'})", sel(cell_type=FOOD_ORNS, side=s), 120.0
            )
        g["ocelli"] = SenseGroup("単眼", sel(cell_sub_class="ocellar"), 80.0)
        self.groups: Dict[str, SenseGroup] = g
        self.values: Dict[str, float] = {k: 0.0 for k in g}
        self._ocelli_prev = None

    def rates(self, obs: Observation, upper_lum: float, dt: float, out: np.ndarray) -> np.ndarray:
        """out（全ニューロンの発火率配列）に感覚入力を書き込む。"""
        v = self.values
        for key in ("sugar", "bitter", "touch_left", "touch_right", "wind", "sound",
                    "odor_left", "odor_right"):
            v[key] = float(np.clip(getattr(obs, key), 0.0, 1.0))
        # 単眼: 画面上部の明るさの変化（暗くなる影 = 捕食者の接近の手がかり）
        if self._ocelli_prev is None:
            self._ocelli_prev = upper_lum
        change = abs(upper_lum - self._ocelli_prev) / max(dt, 1e-3)
        self._ocelli_prev = upper_lum
        v["ocelli"] = float(np.clip(0.05 + 0.5 * change, 0.0, 1.0))
        for key, grp in self.groups.items():
            if len(grp.neurons) and v[key] > 0:
                out[grp.neurons] = np.maximum(out[grp.neurons], grp.max_rate * v[key])
        return out

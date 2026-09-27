"""デモ・テスト用の「トイ・コネクトーム」（人工の小さな回路）。

本物の FlyWire データ（約 135 MB）をダウンロードしなくても全体の流れを
試せるように、同じ注釈の語彙（細胞タイプ名・左右・位置）を持つ約 2,000
ニューロンの回路を人工的に組み立てる。配線は実データで確認した主な経路
（LPLC2/LC4 → 巨大繊維、LC10a → 同側 DNa02、LC9 → P9、糖 → MN9 など）を
簡略化して真似たもので、**本物のハエの脳ではない**。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .connectome import Connectome
from .motor import MN9_ROOT_IDS


def build_toy(seed: int = 0, cols: int = 14, rows: int = 10) -> Connectome:
    rng = np.random.default_rng(seed)
    ann: Dict[str, List[str]] = {k: [] for k in ("super_class", "cell_class", "cell_sub_class", "cell_type", "side")}
    pos: List[Tuple[float, float, float]] = []
    roots: List[int] = []
    groups: Dict[Tuple[str, str], List[int]] = {}

    def add(cell_type, side, super_class, p, cell_class="", sub="", root=None):
        i = len(roots)
        roots.append(root if root is not None else 10**15 + i)
        ann["super_class"].append(super_class)
        ann["cell_class"].append(cell_class)
        ann["cell_sub_class"].append(sub)
        ann["cell_type"].append(cell_type)
        ann["side"].append(side)
        pos.append(p)
        groups.setdefault((cell_type, side), []).append(i)
        return i

    pre, post, w = [], [], []

    def link(a, b, weight):
        pre.append(a)
        post.append(b)
        w.append(weight)

    sides = (("left", -1.0), ("right", 1.0))
    column_of: Dict[int, Tuple[float, float]] = {}  # Tm → (az 0..1, el 0..1)
    tm_by_side: Dict[str, List[int]] = {"left": [], "right": []}
    for side, sgn in sides:
        xc = 450 + sgn * 330
        for u in range(cols):  # u: 0 = 前方, 1 = 後方
            for v in range(rows):  # v: 0 = 背側
                az, el = u / (cols - 1), v / (rows - 1)
                # 髄質（R7/R8）: 視交叉で前後反転 → 前方視野は z が大きい
                pm = (xc + sgn * 10 * rng.random(), 150 + el * 150, 250 - az * 150)
                r7 = add("R7", side, "sensory", pm, "visual")
                r8 = add("R8", side, "sensory", pm, "visual")
                # 視葉板（R1-6）: 前方視野は z が小さい
                pl = (xc + sgn * 60, 150 + el * 150, 100 + az * 150)
                r16 = add("R1-6", side, "sensory", pl, "visual")
                tm = add("Tm1", side, "optic", (xc - sgn * 40, 150 + el * 150, 250 - az * 150))
                column_of[tm] = (az, el)
                tm_by_side[side].append(tm)
                for r in (r7, r8, r16):
                    link(r, tm, 6)

    def vpn(cell_type, side, n, radius, feature_targets):
        out = []
        tms = tm_by_side[side]
        for k in range(n):
            az0, el0 = (k + 0.5) / n, rng.random()
            xc = 450 + (-1 if side == "left" else 1) * 230
            i = add(cell_type, side, "visual_projection", (xc, 200 + 50 * rng.random(), 200 + 50 * rng.random()))
            for tm in tms:
                az, el = column_of[tm]
                if (az - az0) ** 2 + ((el - el0) * 0.7) ** 2 < radius ** 2:
                    link(tm, i, 3)
            out.append(i)
        return out

    vp: Dict[Tuple[str, str], List[int]] = {}
    for side, _ in sides:
        vp[("LPLC2", side)] = vpn("LPLC2", side, 16, 0.25, None)
        vp[("LC4", side)] = vpn("LC4", side, 10, 0.2, None)
        vp[("LC10a", side)] = vpn("LC10a", side, 16, 0.15, None)
        vp[("LC9", side)] = vpn("LC9", side, 10, 0.2, None)
        vp[("LC17", side)] = vpn("LC17", side, 10, 0.18, None)
        vp[("LLPC1", side)] = vpn("LLPC1", side, 10, 0.2, None)
        vp[("LPC1", side)] = vpn("LPC1", side, 10, 0.2, None)
        vp[("MeTu1", side)] = vpn("MeTu1", side, 10, 0.12, None)
        for t in ("HSN", "HSE", "HSS", "VS1", "VS2"):
            vp[(t, side)] = vpn(t, side, 1, 1.5, None)

    dn: Dict[Tuple[str, str], List[int]] = {}
    for side, sgn in sides:
        x = 450 + sgn * 40
        for t, k in (("DNp09", 1), ("DNg100", 1), ("MDN", 2), ("DNa01", 1), ("DNa02", 1),
                     ("DNp01", 1), ("DNp02", 1), ("DNp04", 1), ("DNp11", 1)):
            dn[(t, side)] = [add(t, side, "descending", (x, 380 + 20 * rng.random(), 200)) for _ in range(k)]
    mn9 = [add("CB0701", s, "motor", (450 + g * 20, 420, 150), "brain_motor_neuron", root=r)
           for (s, g), r in zip((("right", 1), ("left", -1)), MN9_ROOT_IDS)]

    def sens(n, side, cls, sub, ct, x):
        return [add(ct, side, "sensory", (x, 330 + 30 * rng.random(), 120), cls, sub) for _ in range(n)]

    sugar = sens(6, "left", "gustatory", "sugar/water", "LB3", 430) + sens(6, "right", "gustatory", "sugar/water", "LB3", 470)
    bitter = sens(4, "left", "gustatory", "bitter", "LB1e", 430) + sens(4, "right", "gustatory", "bitter", "LB1e", 470)
    touch = {s: sens(8, s, "mechanosensory", "head bristle", "BM_Ant", 450 + g * 90) for s, g in sides}
    wind = sens(6, "left", "mechanosensory", "wind_gravity", "JO-EV1", 380) + sens(6, "right", "mechanosensory", "wind_gravity", "JO-EV1", 520)
    sound = sens(6, "left", "mechanosensory", "auditory", "JO-B1_a", 380)
    odor = {s: sens(6, s, "olfactory", "", "ORN_DM1", 450 + g * 60) for s, g in sides}
    ocelli = sens(4, "center", "visual", "ocellar", "", 450)

    # 中枢の介在ニューロン（ランダムな弱い結合で「背景」を作る）
    central = [add(f"CX{k % 20:02d}", "left" if k % 2 else "right", "central",
                   (300 + 300 * rng.random(), 150 + 200 * rng.random(), 100 + 150 * rng.random()))
               for k in range(600)]
    cen = np.array(central)
    for _ in range(4000):
        a, b = rng.choice(cen, 2)
        link(int(a), int(b), int(rng.choice([-4, -2, 2, 3])))

    def all_(key):
        return dn[key]

    opp = {"left": "right", "right": "left"}
    for side, _ in sides:
        o = opp[side]
        for i in vp[("LPLC2", side)]:
            link(i, dn[("DNp01", side)][0], 5)
            link(i, dn[("DNp04", side)][0], 4)
        for i in vp[("LC4", side)]:
            for t in ("DNp01", "DNp02", "DNp04", "DNp11"):
                link(i, dn[(t, side)][0], 5)
        for i in vp[("LC10a", side)]:
            link(i, dn[("DNa02", side)][0], 5)  # 物体の方へ（同側）旋回
        for i in vp[("LLPC1", side)]:
            link(i, dn[("DNa02", side)][0], 3)
        for i in vp[("LC17", side)]:
            link(i, dn[("DNa01", o)][0], 3)
        for i in vp[("LC9", side)]:
            link(i, dn[("DNp09", side)][0], 4)
        for i in odor[side]:
            link(i, dn[("DNa02", side)][0], 6)  # 匂いの方へ
        for i in touch[side]:
            link(i, dn[("DNa02", o)][0], 8)  # 触れた側と反対へ
            for m in dn[("MDN", side)]:
                link(i, m, 4)
        for i in bitter:
            for m in dn[("MDN", side)]:
                link(i, m, 6)
            link(i, dn[("DNp09", side)][0], -8)
        for i in sugar:
            link(i, dn[("DNp09", side)][0], -6)  # 甘いものに触れたら止まる
    for i in sugar:
        for m in mn9:
            link(i, m, 12)
    for i in bitter:
        for m in mn9:
            link(i, m, -20)
    for i in wind + sound + ocelli:
        for c in rng.choice(cen, 5):
            link(i, int(c), 4)
    for c in rng.choice(cen, 40):
        for key in (("DNa02", "left"), ("DNa02", "right"), ("DNp09", "left"), ("DNp09", "right")):
            link(int(c), dn[key][0], int(rng.choice([-3, 3])))

    annotations = {k: np.array(v, dtype=str) for k, v in ann.items()}
    return Connectome.from_edges(np.array(pre), np.array(post), np.array(w, dtype=np.float32),
                                 np.array(roots, dtype=np.int64), ann=annotations,
                                 pos=np.array(pos, dtype=np.float32), name="トイ・コネクトーム（人工回路）")

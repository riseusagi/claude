"""コネクトームから「各ニューロンが視野のどこを見ているか」を推定する。

1. 視細胞 (R1-6: 視葉板 / R7, R8: 髄質) の終末位置を主成分分析し、
   背腹軸と前後軸を取り出して視野の方向 (方位 φ, 仰角 θ) に対応付ける。
   * 背側 = FlyWire の y が小さい側（複眼背縁 DRA が上に来ることを検証済み）
   * 前後: 視葉板は網膜と同じ向き、髄質は第一視交叉で前後が反転する
2. 視覚系ニューロン（optic / visual_projection）について、入力元の方向を
   シナプス数で重み付け平均する操作を繰り返し、方向を下流へ伝播させる。
3. 平均化で分布が視野中央へ縮むのを補正するため、多数の細胞で視野を
   タイル状に覆う細胞タイプについては、同じ眼の視細胞分布に分位点を合わせる。

座標系: φ は正面 0°・右が正、θ は水平 0°・上が正（度）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .connectome import Connectome

VISUAL_CLASSES = ("optic", "visual_projection")

# 片眼の視野（度）。方位は「正面から同側への角度」。正面側は両眼視野のぶん反対側まで少しはみ出す。
EYE_AZ_RANGE = (-15.0, 160.0)
EYE_EL_RANGE = (-60.0, 70.0)


@dataclass
class Retinotopy:
    phi: np.ndarray  # (N,) 度, 不明は nan
    theta: np.ndarray  # (N,) 度, 不明は nan

    @property
    def known(self) -> np.ndarray:
        return ~np.isnan(self.phi)

    def save(self, path) -> None:
        np.savez_compressed(path, phi=self.phi, theta=self.theta)

    @classmethod
    def load(cls, path) -> "Retinotopy":
        with np.load(path) as z:
            return cls(phi=z["phi"], theta=z["theta"])


def _unit(phi_deg: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    p, t = np.radians(phi_deg), np.radians(theta_deg)
    return np.stack([np.cos(t) * np.sin(p), np.sin(t), np.cos(t) * np.cos(p)], axis=-1)


def _angles(v: np.ndarray):
    phi = np.degrees(np.arctan2(v[..., 0], v[..., 2]))
    theta = np.degrees(np.arcsin(np.clip(v[..., 1], -1.0, 1.0)))
    return phi, theta


def _rescale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a, b = np.percentile(x, [2, 98])
    if b - a < 1e-9:
        return np.full_like(x, (lo + hi) / 2)
    return np.clip(lo + (x - a) / (b - a) * (hi - lo), min(lo, hi) - 10, max(lo, hi) + 10)


def photoreceptor_directions(con: Connectome):
    """視細胞のインデックスと視野方向 (φ, θ) を返す。"""
    out_idx, out_phi, out_theta = [], [], []
    for side, sign in (("left", -1.0), ("right", 1.0)):
        for types, medulla in ((["R7", "R8"], True), (["R1-6"], False)):
            idx = con.select(cell_type=types, side=side)
            if len(idx) < 3:
                continue
            P = con.pos[idx].astype(np.float64)
            X = P - P.mean(0)
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            k = int(np.argmax(np.abs(vt[:2, 1])))  # y 成分が大きい方が背腹軸
            dv, ap = vt[k], vt[1 - k]
            if dv[1] > 0:  # +dv を背側（y 小）に
                dv = -dv
            # +ap を「視野の前方」に: 髄質は z 大が前方（視交叉で反転）、視葉板は z 小が前方
            if (ap[2] < 0) == medulla:
                ap = -ap
            el = _rescale(X @ dv, *EYE_EL_RANGE)
            az = _rescale(X @ ap, EYE_AZ_RANGE[1], EYE_AZ_RANGE[0])
            out_idx.append(idx)
            out_phi.append(sign * az)
            out_theta.append(el)
    if not out_idx:
        z = np.zeros(0)
        return z.astype(np.int64), z, z
    return np.concatenate(out_idx), np.concatenate(out_phi), np.concatenate(out_theta)


def infer(con: Connectome, min_input: float = 3.0, iters: int = 12, min_group: int = 20) -> Retinotopy:
    """視覚系全体の網膜位相を推定する（全脳で数秒）。"""
    import scipy.sparse as sp

    n = con.n
    vec = np.full((n, 3), np.nan)
    pr_idx, pr_phi, pr_theta = photoreceptor_directions(con)
    vec[pr_idx] = _unit(pr_phi, pr_theta)

    vis = np.isin(con.ann["super_class"], VISUAL_CLASSES)
    pre = np.repeat(np.arange(n), np.diff(con.indptr))
    post = con.indices.astype(np.int64)
    m = vis[post]
    M = sp.csr_matrix((np.abs(con.weights[m]).astype(np.float64), (post[m], pre[m])), shape=(n, n))

    seed = ~np.isnan(vec[:, 0])
    known = seed.copy()
    for _ in range(iters):
        X = np.where(known[:, None], vec, 0.0)
        num = M @ X
        den = M @ known.astype(np.float64)
        new = (den >= min_input) & ~seed
        norm = np.linalg.norm(num[new], axis=1, keepdims=True)
        ok = norm[:, 0] > 1e-9
        tgt = np.flatnonzero(new)[ok]
        vec[tgt] = num[new][ok] / norm[ok]
        prev = known.sum()
        known = seed.copy()
        known[tgt] = True
        if known.sum() == prev and _ > 2:
            break

    phi, theta = _angles(vec)
    phi = phi.astype(np.float32)
    theta = theta.astype(np.float32)

    # 分位点マッチング（多数の細胞で視野を覆うタイプのみ）
    pr_side = con.ann["side"][pr_idx]
    ct, side = con.ann["cell_type"], con.ann["side"]
    for s in ("left", "right"):
        ref_phi = np.sort(pr_phi[pr_side == s])
        ref_theta = np.sort(pr_theta[pr_side == s])
        if len(ref_phi) < min_group:
            continue
        grp = np.flatnonzero(vis & (side == s) & ~np.isnan(phi) & (ct != ""))
        types, inv = np.unique(ct[grp], return_inverse=True)
        for k, t in enumerate(types):
            members = grp[inv == k]
            if len(members) < min_group:
                continue
            q = (np.argsort(np.argsort(phi[members])) + 0.5) / len(members)
            phi[members] = np.quantile(ref_phi, q)
            q = (np.argsort(np.argsort(theta[members])) + 0.5) / len(members)
            theta[members] = np.quantile(ref_theta, q)
    return Retinotopy(phi=phi, theta=theta)


def load_or_infer(con: Connectome, cache_dir: Optional[Path] = None) -> Retinotopy:
    if cache_dir is None:
        return infer(con)
    key = f"retinotopy_v1_{con.n}_{con.n_connections}.npz"
    path = Path(cache_dir) / key
    if path.exists():
        try:
            return Retinotopy.load(path)
        except Exception:
            pass
    r = infer(con)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        r.save(path)
    except OSError:
        pass
    return r

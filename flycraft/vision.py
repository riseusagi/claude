"""複眼と視葉のエミュレーション。

LIF の全脳モデルでは、視細胞を直接駆動しても信号は視葉の 2〜3 シナプス先で
消えてしまう（視葉の計算は非スパイクの段階的電位と精密な時間特性に依存するため）。
そこで視葉の「よく知られた機能」をソフトウェアで再現し、中枢脳の入口である
視覚投射ニューロン (VPN: LC, LPLC, HS/VS など) をコネクトーム上で駆動する。

処理の流れ:

    ゲーム画面 (RGB)
      → 複眼格子（既定 3° 間隔。実際のハエは約 5°）に平均化して輝度を得る
      → 視葉: ハッセンシュタイン・ライヒャルト型運動検出器 (T4/T5 相当) で
        水平・垂直の局所運動、ON/OFF 変化、小物体運動（中心-周辺差）を計算
      → 各 VPN の受容野（コネクトームから推定した方向とタイプ別の広さ）で
        特徴を集めて発火率に変換
        - LPLC2 / LC4 / LPLC1 / LC16 / LC6: 受容野中心から外向きの運動（ルーミング）
        - LC10 / LC11 / LC12 / LC17 / LC18 など: 小物体の運動
        - HS / LLPC: 同側眼の前→後運動, LPC / H2: 後→前, VS: 下向き運動
        - MeTu: 輝度（天空・偏光経路）, その他: 明暗変化
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .connectome import Connectome
from .retinotopy import Retinotopy

# --------------------------------------------------------------------- camera


@dataclass
class Camera:
    """ゲーム画面の投影（透視投影）。fov_v は垂直視野角（Minecraft の FOV 設定と同じ意味）。"""

    width: int
    height: int
    fov_v: float = 70.0

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)

    @property
    def fov_h(self) -> float:
        return math.degrees(2 * math.atan(math.tan(math.radians(self.fov_v) / 2) * self.aspect))

    def key(self):
        return (self.width, self.height, round(self.fov_v, 3))


def block_mean(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """画像を rows×cols のブロックに分けて平均する（面積平均の縮小）。"""
    h, w = img.shape[:2]
    r_idx = np.linspace(0, h, rows + 1).astype(int)[:-1]
    c_idx = np.linspace(0, w, cols + 1).astype(int)[:-1]
    r_len = np.diff(np.append(r_idx, h))
    c_len = np.diff(np.append(c_idx, w))
    f = img.astype(np.float32)
    s = np.add.reduceat(np.add.reduceat(f, r_idx, axis=0), c_idx, axis=1)
    denom = (r_len[:, None] * c_len[None, :]).astype(np.float32)
    if s.ndim == 3:
        denom = denom[..., None]
    return s / denom


# ------------------------------------------------------------------------ eye


class CompoundEye:
    """画面を複眼の格子にサンプリングする。

    格子は画面上で一様、各格子点の視野方向 (φ, θ) は透視投影から厳密に計算する。
    """

    def __init__(self, camera: Camera, acuity_deg: float = 3.0) -> None:
        self.camera = camera
        self.acuity = acuity_deg
        fh, fv = camera.fov_h, camera.fov_v
        self.cols = max(8, int(round(fh / acuity_deg)))
        self.rows = max(6, int(round(fv / acuity_deg)))
        th, tv = math.tan(math.radians(fh) / 2), math.tan(math.radians(fv) / 2)
        x = (np.arange(self.cols) + 0.5) / self.cols * 2 - 1  # -1..1（右が正）
        y = 1 - (np.arange(self.rows) + 0.5) / self.rows * 2  # 1..-1（上が正）
        X, Y = np.meshgrid(x, y)
        # 方向ベクトル (x, y, z=1) → 角度
        dx, dy = X * th, Y * tv
        self.phi = np.degrees(np.arctan2(dx, 1.0)).astype(np.float32)
        self.theta = np.degrees(np.arctan2(dy, np.sqrt(1 + dx**2))).astype(np.float32)
        # 格子点 1 つが覆う立体角（度²）
        cell = (2 * th / self.cols) * (2 * tv / self.rows)
        self.area = (np.degrees(1) ** 2 * cell / (1 + dx**2 + dy**2) ** 1.5).astype(np.float32)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def sample(self, frame: np.ndarray):
        """RGB フレーム (H,W,3 uint8) → (輝度 [0,1] の格子, RGB 格子 uint8)。"""
        rgb = block_mean(frame[..., :3], self.rows, self.cols)
        # ハエの R1-6 は青緑に感度が高い
        lum = (0.15 * rgb[..., 0] + 0.55 * rgb[..., 1] + 0.30 * rgb[..., 2]) / 255.0
        return lum.astype(np.float32), np.clip(rgb, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- optic lobe


def _gauss_blur(x: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0.3:
        return x
    r = int(max(1, round(2.5 * sigma_px)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_px) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (r, r)), mode="edge")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, pad)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, tmp)


class OpticLobe:
    """視葉の機能的エミュレーション（格子上の特徴マップを計算）。"""

    def __init__(self, shape: Tuple[int, int], acuity_deg: float, tau_emd: float = 0.04) -> None:
        self.shape = shape
        self.acuity = acuity_deg
        self.tau_emd = tau_emd  # T4/T5 の遅延フィルタ時定数 [s]
        self.reset()

    def reset(self) -> None:
        self.c_lp = None
        self.L_prev = None
        self.L_slow = None
        self.features: Dict[str, np.ndarray] = {}

    def update(self, L: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
        dt = max(1e-3, float(dt))
        mean = float(L.mean())
        c = (L - mean) / (mean + 0.05)  # 明順応したコントラスト
        if self.c_lp is None or self.c_lp.shape != c.shape:
            self.c_lp = c.copy()
            self.L_prev = L.copy()
            self.L_slow = L.copy()
        a = dt / (self.tau_emd + dt)
        lp_prev = self.c_lp
        # HR 相関器: 遅延(低域通過)させた隣の信号と現在の信号の積の差
        # 水平: 正 = 右向きの運動
        h = lp_prev[:, :-1] * c[:, 1:] - c[:, :-1] * lp_prev[:, 1:]
        # 垂直: 行は上から下。正 = 上向きの運動（下の行 → 上の行）
        v = lp_prev[1:, :] * c[:-1, :] - c[1:, :] * lp_prev[:-1, :]
        hx = np.zeros_like(c)
        hx[:, :-1] += 0.5 * h
        hx[:, 1:] += 0.5 * h
        vy = np.zeros_like(c)
        vy[:-1, :] += 0.5 * v
        vy[1:, :] += 0.5 * v
        # 1 秒あたりに正規化（フレームレートに依存しにくくする）
        hx /= dt
        vy /= dt
        self.c_lp = lp_prev + a * (c - lp_prev)

        dL = L - self.L_prev
        self.L_prev = L.copy()
        b = dt / (0.3 + dt)
        self.L_slow = self.L_slow + b * (L - self.L_slow)
        on = np.maximum(0.0, L - self.L_slow)
        off = np.maximum(0.0, self.L_slow - L)
        change = np.abs(dL) / dt

        mot = np.sqrt(hx**2 + vy**2)
        surround = _gauss_blur(mot, 15.0 / self.acuity)
        obj = np.maximum(0.0, mot - 1.5 * surround)

        self.features = {
            "lum": L,
            "on": on,
            "off": off,
            "change": change,
            "hx": hx,
            "vy": vy,
            "obj": obj,
        }
        return self.features


# ------------------------------------------------------ visual projection map

# (細胞タイプの正規表現, 特徴, 受容野の σ[度], 閾値 z, 飽和 z, 最大発火率 Hz)
VPN_TUNING: List[Tuple[str, str, float, float, float, float]] = [
    (r"^LPLC2$", "loom", 22.0, 1.3, 3.0, 200.0),
    (r"^LC4$", "loom", 16.0, 1.3, 3.0, 200.0),
    (r"^LPLC1$", "loom", 15.0, 1.3, 3.0, 150.0),
    (r"^LC16$", "loom", 15.0, 1.3, 3.0, 150.0),
    (r"^LC6$", "loom", 18.0, 1.3, 3.0, 120.0),
    (r"^LC(9|10|11|12|13|15|17|18|20|21|22|24|25|26)", "obj", 12.0, 0.6, 2.5, 120.0),
    (r"^LLPC", "ftb", 15.0, 0.6, 2.5, 120.0),
    (r"^LPC", "btf", 15.0, 0.6, 2.5, 120.0),
    (r"^HS[NES]$", "ftb", 50.0, 0.5, 2.0, 150.0),
    (r"^H2$", "btf", 50.0, 0.5, 2.0, 150.0),
    (r"^VS\d", "down", 30.0, 0.5, 2.0, 150.0),
    (r"^MeTu", "lum", 10.0, 0.5, 1.5, 60.0),
    (r"^LC", "obj", 15.0, 0.6, 2.5, 100.0),
    (r".*", "change", 15.0, 0.8, 3.0, 60.0),
]

# 自動利得制御の下限（無刺激時にノイズを増幅しないため）。
# 単位はプーリング後の特徴量（内蔵ワールドで歩行中の上位 10% がおよそ 0.2〜0.5）。
FEATURE_FLOOR = {
    "loom": 0.12,
    "obj": 0.12,
    "ftb": 0.12,
    "btf": 0.12,
    "down": 0.06,
    "lum": 0.2,
    "change": 0.3,
}


@dataclass
class _Group:
    feature: str
    neurons: np.ndarray  # connectome index
    pool: object  # scipy.sparse (n, G)
    sectors: Optional[list] = None  # ルーミング用: 上・下・左・右の各扇形の外向き運動の集計行列
    sign: Optional[np.ndarray] = None  # ftb/btf 用: 右眼 +1, 左眼 -1
    thr: np.ndarray = None
    sat: np.ndarray = None
    rmax: np.ndarray = None


class VisualProjection:
    """複眼格子の特徴 → VPN の発火率（Hz）。"""

    def __init__(self, con: Connectome, retino: Retinotopy, eye: CompoundEye, agc_tau: float = 4.0):
        import scipy.sparse as sp

        self.eye = eye
        self.agc_tau = agc_tau
        self.scale: Dict[str, float] = {}
        vpn = np.flatnonzero((con.ann["super_class"] == "visual_projection") & retino.known)
        types = con.ann["cell_type"][vpn]
        side = con.ann["side"][vpn]
        rules = [(re.compile(p), f, s, t, z, r) for p, f, s, t, z, r in VPN_TUNING]
        assign: Dict[str, list] = {}
        for i, (nid, t) in enumerate(zip(vpn, types)):
            for rx, feat, sigma, thr, sat, rmax in rules:
                if rx.search(t or ""):
                    assign.setdefault(feat, []).append((nid, sigma, thr, sat, rmax, side[i]))
                    break

        gphi = eye.phi.ravel().astype(np.float64)
        gth = eye.theta.ravel().astype(np.float64)
        gvec = _unit(gphi, gth)
        area = eye.area.ravel().astype(np.float64)
        self.groups: List[_Group] = []
        for feat, items in assign.items():
            rows, cols, vals, kept = [], [], [], []
            sec = [[], [], [], []]  # 各扇形の重み（外向き成分の係数を含む）
            for nid, sigma, thr, sat, rmax, sd in items:
                c = _unit(np.array([retino.phi[nid]]), np.array([retino.theta[nid]]))[0]
                cosd = np.clip(gvec @ c, -1.0, 1.0)
                d = np.degrees(np.arccos(cosd))
                m = d < 2.5 * sigma
                if not m.any():
                    continue
                w = np.exp(-0.5 * (d[m] / sigma) ** 2) * area[m] / (2 * math.pi * sigma**2)
                r = len(kept)
                kept.append((nid, thr, sat, rmax, 1.0 if sd == "right" else -1.0))
                idx = np.flatnonzero(m)
                rows.append(np.full(len(idx), r))
                cols.append(idx)
                vals.append(w)
                if feat == "loom":
                    # 受容野中心から見た各格子点の向き（接平面上）で 4 つの扇形に分け、
                    # それぞれ外向きの運動成分（上: +vy, 下: -vy, 右: +hx, 左: -hx）を集める。
                    dphi = (gphi[idx] - retino.phi[nid] + 180) % 360 - 180
                    ux = dphi * math.cos(math.radians(retino.theta[nid]))
                    uy = gth[idx] - retino.theta[nid]
                    up, down = (uy > np.abs(ux)), (-uy > np.abs(ux))
                    right, left = (ux >= np.abs(uy)), (-ux >= np.abs(uy))
                    for k, m_ in enumerate((up, down, right, left)):
                        sec[k].append(w * m_ * 4.0)
            if not kept:
                continue
            G = gvec.shape[0]
            R = np.concatenate(rows)
            C = np.concatenate(cols)
            shape = (len(kept), G)
            grp = _Group(
                feature=feat,
                neurons=np.array([k[0] for k in kept], dtype=np.int64),
                pool=sp.csr_matrix((np.concatenate(vals), (R, C)), shape=shape),
                thr=np.array([k[1] for k in kept], dtype=np.float32),
                sat=np.array([k[2] for k in kept], dtype=np.float32),
                rmax=np.array([k[3] for k in kept], dtype=np.float32),
                sign=np.array([k[4] for k in kept], dtype=np.float32),
            )
            if feat == "loom":
                grp.sectors = [sp.csr_matrix((np.concatenate(v), (R, C)), shape=shape) for v in sec]
            self.groups.append(grp)
        self.n_driven = int(sum(len(g.neurons) for g in self.groups))
        self.last: Dict[str, np.ndarray] = {}

    def rates(self, feats: Dict[str, np.ndarray], dt: float, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        a = min(1.0, dt / self.agc_tau)
        for g in self.groups:
            f = g.feature
            if f == "loom":
                # LPLC2 型: 4 方向すべてで外向き運動があるときだけ応答する（幾何平均）。
                # 歩行による地面の流れ（並進）は一部の扇形で内向きになるので打ち消される。
                hx, vy = feats["hx"].ravel(), feats["vy"].ravel()
                up = np.maximum(g.sectors[0] @ vy, 0.0)
                down = np.maximum(-(g.sectors[1] @ vy), 0.0)
                right = np.maximum(g.sectors[2] @ hx, 0.0)
                left = np.maximum(-(g.sectors[3] @ hx), 0.0)
                x = (up * down * right * left) ** 0.25
            elif f in ("ftb", "btf"):
                x = (g.pool @ feats["hx"].ravel()) * g.sign
                if f == "btf":
                    x = -x
            elif f == "down":
                x = -(g.pool @ feats["vy"].ravel())
            else:
                x = g.pool @ feats[f].ravel()
            # 受容野の大きさで正規化した値 → 自動利得制御
            x = np.asarray(x, dtype=np.float32)
            pos = np.maximum(x, 0.0)
            ref = float(np.percentile(pos, 90)) if len(pos) else 0.0
            s = self.scale.get(f, max(ref, FEATURE_FLOOR[f]))
            s = s + a * (ref - s)
            s = max(s, FEATURE_FLOOR[f])
            self.scale[f] = s
            z = pos / s
            r = g.rmax * np.clip((z - g.thr) / (g.sat - g.thr), 0.0, 1.0)
            out[g.neurons] = r
            self.last[f] = r
        return out


def _unit(phi_deg, theta_deg):
    p, t = np.radians(phi_deg), np.radians(theta_deg)
    return np.stack([np.cos(t) * np.sin(p), np.sin(t), np.cos(t) * np.cos(p)], axis=-1)

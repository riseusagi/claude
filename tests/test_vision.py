import numpy as np

from flycraft.retinotopy import Retinotopy
from flycraft.toy import build_toy
from flycraft.retinotopy import infer
from flycraft.vision import Camera, CompoundEye, OpticLobe, VisualProjection


def grating(w, h, phase, period=12):
    x = np.arange(w)
    row = (0.5 + 0.4 * np.sin(2 * np.pi * (x - phase) / period)) * 255
    img = np.repeat(row[None, :], h, axis=0)
    return np.stack([img] * 3, -1).astype(np.uint8)


def test_compound_eye_geometry():
    eye = CompoundEye(Camera(160, 90, 70.0), 3.0)
    assert eye.phi[:, 0].mean() < -40 and eye.phi[:, -1].mean() > 40
    assert eye.theta[0].mean() > 25 and eye.theta[-1].mean() < -25
    assert eye.area.min() > 0


def test_emd_detects_direction():
    eye = CompoundEye(Camera(96, 60, 60.0), 3.0)
    lobe = OpticLobe(eye.shape, 3.0)
    hx = []
    for t in range(12):
        L, _ = eye.sample(grating(96, 60, phase=2 * t))  # 右へ動く縞
        f = lobe.update(L, 0.05)
        if t > 3:
            hx.append(f["hx"].mean())
    assert np.mean(hx) > 0
    lobe.reset()
    hx = []
    for t in range(12):
        L, _ = eye.sample(grating(96, 60, phase=-2 * t))  # 左へ
        f = lobe.update(L, 0.05)
        if t > 3:
            hx.append(f["hx"].mean())
    assert np.mean(hx) < 0


def _square(w, h, size):
    img = np.full((h, w, 3), 200, np.uint8)
    cy, cx = h // 2, w // 2
    img[max(0, cy - size):cy + size, max(0, cx - size):cx + size] = 20
    return img


def test_lplc2_like_loom_detector():
    """中央で広がる黒い四角（ルーミング）には応答し、横に動く縞（並進）には応答しない。"""
    con = build_toy()
    # 正面を向いた LPLC2 を 1 つだけ作るための簡易網膜位相
    phi = np.full(con.n, np.nan, np.float32)
    theta = np.full(con.n, np.nan, np.float32)
    lp = con.select(cell_type="LPLC2", side="right")[0]
    phi[lp], theta[lp] = 0.0, 0.0
    eye = CompoundEye(Camera(120, 90, 80.0), 3.0)
    vp = VisualProjection(con, Retinotopy(phi, theta), eye)
    lobe = OpticLobe(eye.shape, 3.0)
    loom = []
    for t in range(14):
        L, _ = eye.sample(_square(120, 90, 4 + 3 * t))
        r = vp.rates(lobe.update(L, 0.05), 0.05, con.n)
        loom.append(r[lp])
    assert max(loom) > 50
    lobe.reset()
    vp.scale.clear()
    trans = []
    for t in range(14):
        L, _ = eye.sample(grating(120, 90, phase=3 * t))
        r = vp.rates(lobe.update(L, 0.05), 0.05, con.n)
        trans.append(r[lp])
    assert max(trans) < 5


def test_toy_retinotopy_sides():
    con = build_toy()
    r = infer(con)
    for t in ("LPLC2", "LC10a"):
        left = con.select(cell_type=t, side="left")
        right = con.select(cell_type=t, side="right")
        assert np.nanmean(r.phi[left]) < -5
        assert np.nanmean(r.phi[right]) > 5
    # 背側（v=0）の視細胞は上を向く
    pr = con.select(cell_type="R7", side="left")
    y = con.pos[pr, 1]
    assert np.corrcoef(y, r.theta[pr])[0, 1] < -0.9

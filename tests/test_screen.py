"""画面モード（方法 B）のテスト。内蔵ワールドを「画面」と「キーボード・マウス」に見立てて動かす。"""

import math
import time

import numpy as np

from flycraft.backends.screen import (Controller, KeyPolicy, ScreenBackend, _NullInput, parallax_residual,
                                      phase_shift, window_score)
from flycraft.backends.sim import STONE, SimWorld
from flycraft.interface import Action

DPP = 0.15  # 疑似ゲームのマウス感度 [度/px]


class SimScreen:
    """SimWorld を描画する疑似キャプチャ + それを操作する疑似キーボード・マウス。"""

    def __init__(self, seed=0):
        self.world = SimWorld(seed=seed, width=192, height=108, n_slimes=0)
        self.world.reset()
        self.hwnd, self.region, self.method = None, None, "fake"
        self.keys = set()
        self.log = []
        self.cursor = False  # True = カーソル表示（メニュー中）
        self.menu = False  # True = メニュー・チャット中（マウスで視点が回らない）

    # capture API
    def start(self):
        pass

    def locate(self):
        return (0, 0, 192, 108)

    def latest(self):
        return self.world.render(), time.time()

    def wait_frame(self, after=0.0, timeout=2.0):
        return self.world.render()

    def grab(self):
        return self.world.render()

    def foreground(self):
        return True

    def close(self):
        pass

    # input API
    def key(self, name, down):
        self.log.append(("key", name, down))
        (self.keys.add if down else self.keys.discard)(name)

    def move(self, dx, dy):
        self.log.append(("move", dx, dy))
        if self.menu:
            return
        p = self.world.player
        p.yaw = (p.yaw - dx * DPP) % 360
        p.pitch = float(np.clip(p.pitch - dy * DPP, -90, 90))

    def button(self, down):
        self.log.append(("button", down))

    def hotkey_pressed(self, vk):
        return False

    def tick(self, dt=0.05):
        f = (1.0 if "w" in self.keys else 0.0) - (1.0 if "s" in self.keys else 0.0)
        self.world.step(Action(forward=f, jump="space" in self.keys), dt)


def make_backend(fake, **kw):
    be = ScreenBackend(capture=fake, input_device=fake, countdown=0, log=lambda *_: None, **kw)
    be.cursor_hidden = lambda: not fake.cursor
    return be


# ------------------------------------------------------------------ controller
def test_forward_hysteresis_and_no_double_tap():
    dev = _NullInput()
    c = Controller(dev, KeyPolicy())
    t = 0.0
    presses = []
    # 閾値付近で揺れる前進指令 → W は小刻みに押し直されない
    for i in range(200):
        f = 0.35 if i % 2 else 0.1
        c.apply(Action(forward=f), t, 0.05)
        t += 0.05
    downs = [e for e in dev.log if e == ("key", "w", True)]
    ups = [i for i, e in enumerate(dev.log) if e == ("key", "w", False)]
    assert len(downs) <= 1 + len(ups)
    # 押下と押下の間隔は min_gap 以上（統合版の 2 度押しダッシュを起こさない）
    dev = _NullInput()
    c = Controller(dev, KeyPolicy())
    t = 0.0
    for i in range(400):
        f = 1.0 if (i // 3) % 2 == 0 else 0.0
        before = len(dev.log)
        c.apply(Action(forward=f), t, 0.05)
        if ("key", "w", True) in dev.log[before:]:
            presses.append(t)
        t += 0.05
    gaps = np.diff(presses)
    assert len(presses) > 3 and gaps.min() >= 0.25 + 0.45 - 1e-9


def test_jump_cooldown_prevents_double_tap():
    dev = _NullInput()
    c = Controller(dev, KeyPolicy())
    t, presses = 0.0, []
    for _ in range(100):
        before = len(dev.log)
        c.apply(Action(jump=True), t, 0.05)
        if ("key", "space", True) in dev.log[before:]:
            presses.append(t)
        t += 0.05
    assert len(presses) >= 5
    assert np.diff(presses).min() >= 0.6 - 1e-9
    assert dev.log.count(("key", "space", False)) >= len(presses) - 1


def test_turn_uses_real_time_and_calibration():
    dev = _NullInput()
    c = Controller(dev, KeyPolicy(turn_deg_s=150), deg_per_px=0.15)
    c.apply(Action(turn=1.0), 0.0, 0.1)  # 左へ 15° → マウス左へ 100 px
    moves = [e for e in dev.log if e[0] == "move"]
    assert moves and sum(m[1] for m in moves) == -100


def test_attack_is_held_briefly():
    dev = _NullInput()
    c = Controller(dev, KeyPolicy())
    c.apply(Action(attack=True), 0.0, 0.05)
    c.apply(Action(attack=False), 0.1, 0.05)
    assert ("button", False) not in dev.log  # すぐには離さない
    c.apply(Action(attack=False), 0.5, 0.05)
    assert ("button", False) in dev.log


def test_window_score_prefers_minecraft_and_skips_browsers():
    assert window_score("Minecraft", "minecraft.windows.exe", "Minecraft") > window_score("Minecraft", "other.exe", "Minecraft")
    assert window_score("Minecraft Wiki - Google Chrome", "chrome.exe", "Minecraft") == 0
    assert window_score("python -m flycraft screen", "python.exe", "Minecraft") == 0
    assert window_score("Minecraft Preview", "minecraft.windows.exe", "Minecraft") >= 2


# ------------------------------------------------------------------- vision
def test_phase_shift_finds_translation():
    rng = np.random.default_rng(0)
    a = rng.random((60, 96)) * 255
    b = np.roll(a, (3, -7), axis=(0, 1))
    dx, dy, peak = phase_shift(a, b)
    assert abs(dx + 7) < 0.6 and abs(dy - 3) < 0.6 and peak > 0.3


def test_parallax_separates_walking_from_pushing_a_wall():
    def residuals(wall):
        w = SimWorld(seed=0, width=192, height=108, n_slimes=0, auto_step=False)
        w.reset()
        p = w.player
        p.yaw, p.pitch = 90.0, -8.0
        if wall:
            x = int(math.floor(p.x + 0.9))
            w.world[x, int(p.y):int(p.y) + 6, int(p.z) - 6:int(p.z) + 7] = STONE
        from flycraft.backends.screen import _gray_small

        out, prev, t = [], None, 0.0
        for _ in range(12):
            for _ in range(3):
                w.step(Action(forward=1.0), 0.04)
                t += 0.04
            w.view_offset = [0, 0.06 * abs(math.sin(2 * math.pi * t)), 0, 0.8 * math.sin(2 * math.pi * t), 0.4]
            g = _gray_small(w.render())
            if prev is not None:
                out.append(parallax_residual(prev, g))
            prev = g
        return np.median(out), w.stats["distance"]

    wall_r, wall_d = residuals(True)
    assert wall_d < 0.5 and wall_r < 0.06


# ------------------------------------------------------------------ backend
def test_calibration_measures_mouse_sensitivity_and_levels_pitch():
    fake = SimScreen(seed=1)
    fake.world.player.pitch = 40.0
    be = make_backend(fake, pitch=8.0)
    be.reset()
    assert abs(be.ctl.deg_per_px - DPP) / DPP < 0.05
    assert abs(fake.world.player.pitch - (-8.0)) < 3.0


def test_no_input_while_menu_is_open():
    fake = SimScreen()
    be = make_backend(fake, calibrate=False)
    be.reset()
    be.step(Action(forward=1.0, attack=True), 0.0)
    assert "w" in fake.keys and ("button", True) in fake.log
    fake.cursor = True  # ポーズメニューが開いた
    n = len(fake.log)
    for _ in range(5):
        be.step(Action(forward=1.0, attack=True, jump=True, turn=1.0), 0.0)
    new = fake.log[n:]
    assert "w" not in fake.keys
    assert all(e[2] is False for e in new if e[0] == "key")  # 離すだけ
    assert all(e == ("button", False) for e in new if e[0] == "button")
    assert not any(e[0] == "move" for e in new)
    assert "カーソル" in be.status


def test_pushing_a_wall_is_felt_as_touch():
    fake = SimScreen(seed=0)
    w = fake.world
    p = w.player
    p.yaw, p.pitch = 90.0, -8.0
    x = int(math.floor(p.x + 0.9))
    w.world[x, int(p.y):int(p.y) + 6, int(p.z) - 6:int(p.z) + 7] = STONE
    w.auto_step = False
    be = make_backend(fake, calibrate=False)
    be.reset()
    touch = 0.0
    t0 = time.time()
    while time.time() - t0 < 2.0:
        o = be.step(Action(forward=1.0), 0.05)
        fake.tick()
        touch = max(touch, o.touch_left)
    assert touch > 0.5


def test_windows_struct_sizes():
    """SendInput / GetCursorInfo の構造体サイズ（64 ビットで間違えると黙って失敗する）。"""
    import ctypes

    from flycraft.backends.screen import win_input_structs

    INPUT, KEYBDINPUT, MOUSEINPUT, CURSORINFO = win_input_structs()
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(INPUT) == 40
        assert ctypes.sizeof(MOUSEINPUT) == 32
        assert ctypes.sizeof(CURSORINFO) == 24
    else:
        assert ctypes.sizeof(INPUT) == 28


def test_calibration_handles_low_and_high_mouse_sensitivity(monkeypatch):
    """マウス感度が低くても高くても（1 px = 0.01〜1.5°）較正できる。"""
    import test_screen as T

    for dpp in (0.01, 0.6, 1.5):
        monkeypatch.setattr(T, "DPP", dpp)
        fake = SimScreen(seed=1)
        be = make_backend(fake, pitch=8.0)
        be.reset()
        assert abs(be.ctl.deg_per_px - dpp) / dpp < 0.05
        assert abs(fake.world.player.pitch - (-8.0)) < 3.0


def test_default_turn_is_fast_enough():
    """旋回指令 0.25（脳でよく出る大きさ）で 45°/秒、1 px = 0.15° なら 1 秒で 300 px 動かす。"""
    dev = _NullInput()
    c = Controller(dev, KeyPolicy(), deg_per_px=0.15)
    t = 0.0
    for _ in range(20):
        c.apply(Action(turn=-0.25), t, 0.05)
        t += 0.05
    px = sum(e[1] for e in dev.log if e[0] == "move")
    assert px == 300


def test_stops_when_mouse_does_not_turn_the_view():
    """統合版のチャット等でカーソル判定が効かなくても、視点が回らなければ入力を止める。"""
    fake = SimScreen(seed=2)
    be = make_backend(fake, calibrate=False)  # カーソル判定は常に「隠れている」
    be.reset()

    def run(seconds, turn=0.0):
        t0 = time.time()
        while time.time() - t0 < seconds:
            be.step(Action(forward=1.0, turn=turn), 0.05)

    run(1.0)
    assert "w" in fake.keys and be.view.ok
    fake.menu = True  # チャットを開いた（旋回していなくても定期的な確認で気づく）
    run(4.0)
    assert not be.view.ok
    assert "w" not in fake.keys
    assert "視点" in be.status
    fake.menu = False  # ゲームに戻った
    run(3.0)
    assert be.view.ok and "w" in fake.keys


def test_view_check_passes_during_normal_turning():
    fake = SimScreen(seed=3)
    be = make_backend(fake, calibrate=False)
    be.reset()
    t0 = time.time()
    while time.time() - t0 < 3.0:
        be.step(Action(forward=0.5, turn=0.4), 0.05)
        assert be.view.ok


def test_screen_turn_rate_matches_setting():
    """脳の計算が速くて待ち時間が長くても、設定どおりの速さで回る（待ち時間を旋回量に含める）。"""
    fake = SimScreen(seed=4)
    be = make_backend(fake, calibrate=False, check_view=False)
    be.reset()
    be.set_turn_speed(360)
    yaw0 = fake.world.player.yaw
    t0 = time.time()
    while time.time() - t0 < 1.0:
        be.step(Action(turn=0.25), 0.05)
    dyaw = ((fake.world.player.yaw - yaw0 + 180) % 360) - 180
    assert 70 < dyaw < 110  # 0.25 × 360°/秒 × 1 秒 = 90°（+ = 左）

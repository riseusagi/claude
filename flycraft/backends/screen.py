"""画面キャプチャ + 仮想キーボード/マウスで Minecraft 統合版を操作する。

* 視覚: Minecraft のウィンドウを mss でキャプチャする（pip install mss）
* 操作: Windows は SendInput（スキャンコード）で W/S/Space/左クリックとマウス移動を送る。
  それ以外の OS では pynput があればそれを使う。
* 安全装置: Minecraft のウィンドウが最前面のときだけ入力を送る。F8 で一時停止/再開。
  一時停止・終了時は押しっぱなしのキーをすべて離す。
* 触覚: W を押しているのに画面がほとんど変わらない → 何かにぶつかっている、とみなす。

統合版の設定で「自動ジャンプ」をオンにしておくと 1 段の段差を登れる。
"""

from __future__ import annotations

import ctypes
import math
import sys
import time
from typing import Callable, Optional, Tuple

import numpy as np

from ..interface import Action, Backend, Observation

IS_WIN = sys.platform == "win32"


# ------------------------------------------------------------------ windows
def _win_dpi_aware() -> None:
    if not IS_WIN:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def find_window(title_part: str):
    """タイトルに title_part を含む表示中のウィンドウ (hwnd) を探す（Windows）。"""
    if not IS_WIN:
        return None
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title_part.lower() in buf.value.lower() and "flycraft" not in buf.value.lower():
                    found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(proc(cb), 0)
    # 完全一致を優先
    found.sort(key=lambda t: (t[1].lower() != title_part.lower(), len(t[1])))
    return found[0][0] if found else None


def client_rect(hwnd) -> Optional[Tuple[int, int, int, int]]:
    if not IS_WIN or not hwnd:
        return None
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    r = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(r)):
        return None
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, r.right - r.left, r.bottom - r.top


def is_foreground(hwnd) -> bool:
    if not IS_WIN or not hwnd:
        return True
    return ctypes.windll.user32.GetForegroundWindow() == hwnd


# ------------------------------------------------------------------ capture
class ScreenCapture:
    def __init__(self, window: Optional[str] = "Minecraft", region: Optional[str] = None,
                 max_width: int = 256) -> None:
        _win_dpi_aware()
        try:
            import mss  # noqa: F401
        except ImportError as e:  # pragma: no cover - 環境依存
            raise RuntimeError("画面キャプチャには mss が必要です: pip install mss") from e
        self.window = window
        self.region = tuple(int(v) for v in region.split(",")) if region else None
        self.max_width = max_width
        self.hwnd = None
        self._sct = None

    def locate(self) -> Tuple[int, int, int, int]:
        if self.region:
            return self.region  # type: ignore[return-value]
        if self.window and IS_WIN:
            if not self.hwnd:
                self.hwnd = find_window(self.window)
            rect = client_rect(self.hwnd) if self.hwnd else None
            if rect and rect[2] > 50 and rect[3] > 50:
                return rect
        import mss

        with mss.mss() as s:
            m = s.monitors[1]
            return m["left"], m["top"], m["width"], m["height"]

    def grab(self) -> np.ndarray:
        import mss

        if self._sct is None:
            self._sct = mss.mss()
        x, y, w, h = self.locate()
        shot = self._sct.grab({"left": x, "top": y, "width": w, "height": h})
        img = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        step = max(1, int(math.ceil(shot.width / self.max_width)))
        return np.ascontiguousarray(img[::step, ::step, 2::-1])  # BGRA → RGB

    @property
    def foreground(self) -> bool:
        return is_foreground(self.hwnd) if self.hwnd else True


# -------------------------------------------------------------------- input
class _WinInput:
    SCAN = {"w": 0x11, "a": 0x1E, "s": 0x1F, "d": 0x20, "space": 0x39, "shift": 0x2A}

    def __init__(self) -> None:
        from ctypes import wintypes

        ULONG_PTR = ctypes.c_size_t

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

        class _U(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self.INPUT, self.KEYBDINPUT, self.MOUSEINPUT = INPUT, KEYBDINPUT, MOUSEINPUT
        self.user32 = ctypes.windll.user32

    def _send(self, inp) -> None:
        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def key(self, name: str, down: bool) -> None:
        flags = 0x0008 | (0 if down else 0x0002)  # SCANCODE | KEYUP
        inp = self.INPUT(type=1)
        inp.u.ki = self.KEYBDINPUT(0, self.SCAN[name], flags, 0, 0)
        self._send(inp)

    def move(self, dx: int, dy: int) -> None:
        inp = self.INPUT(type=0)
        inp.u.mi = self.MOUSEINPUT(int(dx), int(dy), 0, 0x0001, 0, 0)
        self._send(inp)

    def button(self, down: bool) -> None:
        inp = self.INPUT(type=0)
        inp.u.mi = self.MOUSEINPUT(0, 0, 0, 0x0002 if down else 0x0004, 0, 0)
        self._send(inp)

    def pressed(self, vk: int) -> bool:
        return bool(self.user32.GetAsyncKeyState(vk) & 0x0001)


class _PynputInput:  # pragma: no cover - 環境依存
    def __init__(self) -> None:
        from pynput import keyboard, mouse

        self.kb = keyboard.Controller()
        self.ms = mouse.Controller()
        self.K = {"w": "w", "a": "a", "s": "s", "d": "d", "space": keyboard.Key.space,
                  "shift": keyboard.Key.shift}
        self.Button = mouse.Button

    def key(self, name, down):
        (self.kb.press if down else self.kb.release)(self.K[name])

    def move(self, dx, dy):
        self.ms.move(int(dx), int(dy))

    def button(self, down):
        (self.ms.press if down else self.ms.release)(self.Button.left)

    def pressed(self, vk):
        return False


class _NullInput:
    """テスト用: 入力を送らず記録するだけ。"""

    def __init__(self) -> None:
        self.log = []

    def key(self, name, down):
        self.log.append(("key", name, down))

    def move(self, dx, dy):
        self.log.append(("move", dx, dy))

    def button(self, down):
        self.log.append(("button", down))

    def pressed(self, vk):
        return False


def make_input():
    if IS_WIN:
        return _WinInput()
    try:
        return _PynputInput()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("キー入力の送信には Windows か pynput が必要です: pip install pynput") from e


# ------------------------------------------------------------------ backend
class InputState:
    """押しているキーを管理し、変化があったときだけ送る。"""

    def __init__(self, dev) -> None:
        self.dev = dev
        self.down = {"w": False, "s": False, "space": False, "mouse": False}
        self._mx = 0.0

    def set(self, name: str, on: bool) -> None:
        if self.down[name] == on:
            return
        self.down[name] = on
        if name == "mouse":
            self.dev.button(on)
        else:
            self.dev.key(name, on)

    def turn(self, px: float) -> None:
        self._mx += px
        step = int(self._mx)
        if step:
            self._mx -= step
            self.dev.move(step, 0)

    def release_all(self) -> None:
        for k in list(self.down):
            self.set(k, False)


class ScreenBackend(Backend):
    name = "screen"
    VK_F8 = 0x77

    def __init__(self, window: Optional[str] = "Minecraft", region: Optional[str] = None,
                 fov_v: float = 70.0, mouse_speed: float = 6.0, countdown: float = 3.0,
                 capture=None, input_device=None, log: Callable[[str], None] = print) -> None:
        self.cap = capture or ScreenCapture(window, region)
        self.inp = InputState(input_device or make_input())
        self.fov_v = fov_v
        self.mouse_speed = mouse_speed
        self.countdown = countdown
        self.log = log
        self.paused = False
        self._last = None
        self._prev_small = None
        self._still = 0.0
        self._touch = 0.0
        self._t_last = time.perf_counter()

    @property
    def realtime(self) -> bool:
        return True

    def reset(self) -> Observation:
        x, y, w, h = self.cap.locate()
        self.log(f"🎮 キャプチャ範囲: x={x} y={y} {w}×{h}"
                 + ("（ウィンドウが見つからないので画面全体）" if IS_WIN and not self.cap.hwnd and not self.cap.region else ""))
        self.log("   ・Minecraft のウィンドウを最前面にすると操作が始まります（最前面でない間は入力しません）")
        self.log("   ・F8 で一時停止/再開、ターミナルで Ctrl+C で終了")
        self.log("   ・統合版の設定で『自動ジャンプ』をオンにすると段差を登れます")
        for k in range(int(self.countdown), 0, -1):
            self.log(f"   {k}…")
            time.sleep(1.0)
        return self._observe(0.05)

    def _observe(self, dt: float) -> Observation:
        frame = self.cap.grab()
        small = frame[::4, ::4].astype(np.float32).mean(axis=2)
        if self._prev_small is not None and self._prev_small.shape == small.shape:
            diff = float(np.abs(small - self._prev_small).mean())
            walking = self.inp.down["w"]
            if walking and diff < 1.2:
                self._still += dt
            else:
                self._still = 0.0
        self._prev_small = small
        self._touch = 0.8 if self._still > 0.35 else self._touch * math.exp(-dt / 0.3)
        return Observation(frame=frame, fov_v=self.fov_v, touch_left=self._touch, touch_right=self._touch,
                           info={"paused": self.paused, "foreground": self.cap.foreground})

    def step(self, action: Action, dt: float) -> Observation:
        if self.inp.dev.pressed(self.VK_F8):
            self.paused = not self.paused
            self.log("⏸ 一時停止" if self.paused else "▶ 再開")
        active = (not self.paused) and self.cap.foreground
        if active:
            self.inp.set("w", action.forward > 0.25)
            self.inp.set("s", action.forward < -0.25)
            self.inp.set("mouse", bool(action.attack))
            if self.inp.down["space"]:
                self.inp.set("space", False)
            elif action.jump:
                self.inp.set("space", True)
            # 旋回: + = 左 → マウスは左（dx < 0）
            self.inp.turn(-action.turn * self.mouse_speed * (dt / 0.05))
        else:
            self.inp.release_all()
        # 実時間に合わせる
        now = time.perf_counter()
        wait = dt - (now - self._t_last)
        if wait > 0:
            time.sleep(wait)
        self._t_last = time.perf_counter()
        return self._observe(dt)

    def close(self) -> None:
        try:
            self.inp.release_all()
        except Exception:
            pass

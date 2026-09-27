"""画面キャプチャ + 仮想キーボード/マウスで Minecraft 統合版を操作する（Windows 専用）。

* 視覚: Minecraft のウィンドウを別スレッドでキャプチャし続け、最新の画像を使う。
  GDI の StretchBlt（HALFTONE = 面積平均の縮小）で小さく取り込むので軽い。
* 操作: SendInput（スキャンコード + 相対マウス移動）。追加のライブラリは不要（ctypes のみ）。
* 安全装置（入力を送るのは次をすべて満たすときだけ）:
  - Minecraft のウィンドウが最前面
  - マウスカーソルが隠れている（= ゲームがマウスを掴んでいる。ポーズ・インベントリ・チャット中は
    カーソルが出るので、メニューのボタンを誤ってクリックしない）
  - F8 で一時停止していない
  条件が崩れた瞬間に押しているキーをすべて離す。終了時・異常終了時も離す。
* 統合版の癖への対策:
  - W の素早い 2 度押しはダッシュになる → 前進にヒステリシスと再押下までの最小間隔
  - クリエイティブで Space の 2 度押しは飛行の切り替え → ジャンプに 0.6 秒のクールダウン
  - 旋回はマウスの移動量で決まり感度に依存する → 起動時に自動較正（マウスを少し動かして
    画面のずれから「1 px あたり何度回るか」を測る）し、視線の上下も水平付近にそろえる
* 触覚: W を押しているのに画面に視差（奥行きによる動きの違い）が生じない → 壁に当たっている。
  回転や視点の揺れは画面全体の一様なずれなので、ずれを補正した残差で判定する。

統合版の設定: 「自動ジャンプ」オン、「表示の揺れ」オフ推奨。F1 で HUD（手・ホットバー）を隠すと見やすい。
"""

from __future__ import annotations

import atexit
import ctypes
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..interface import Action, Backend, Observation

IS_WIN = sys.platform == "win32"

Rect = Tuple[int, int, int, int]


# ================================================================== Windows
_PROTO_DONE = False


def _win_prototypes() -> None:
    """64 ビット Windows でハンドルが切り詰められないよう、使う API の型を宣言する。"""
    global _PROTO_DONE
    if _PROTO_DONE or not IS_WIN:
        return
    from ctypes import wintypes as W

    u, g, k = ctypes.windll.user32, ctypes.windll.gdi32, ctypes.windll.kernel32
    u.GetForegroundWindow.restype = W.HWND
    u.GetAncestor.argtypes, u.GetAncestor.restype = (W.HWND, W.UINT), W.HWND
    u.IsWindow.argtypes = (W.HWND,)
    u.IsWindowVisible.argtypes = (W.HWND,)
    u.IsIconic.argtypes = (W.HWND,)
    u.GetWindowTextLengthW.argtypes = (W.HWND,)
    u.GetWindowTextW.argtypes = (W.HWND, W.LPWSTR, ctypes.c_int)
    u.GetWindowThreadProcessId.argtypes = (W.HWND, ctypes.POINTER(W.DWORD))
    u.GetClientRect.argtypes = (W.HWND, ctypes.POINTER(W.RECT))
    u.ClientToScreen.argtypes = (W.HWND, ctypes.POINTER(W.POINT))
    u.GetDC.argtypes, u.GetDC.restype = (W.HWND,), W.HDC
    u.ReleaseDC.argtypes = (W.HWND, W.HDC)
    k.GetConsoleWindow.restype = W.HWND
    k.OpenProcess.argtypes, k.OpenProcess.restype = (W.DWORD, W.BOOL, W.DWORD), W.HANDLE
    k.CloseHandle.argtypes = (W.HANDLE,)
    k.QueryFullProcessImageNameW.argtypes = (W.HANDLE, W.DWORD, W.LPWSTR, ctypes.POINTER(W.DWORD))
    g.CreateCompatibleDC.argtypes, g.CreateCompatibleDC.restype = (W.HDC,), W.HDC
    g.CreateDIBSection.argtypes = (W.HDC, ctypes.c_void_p, W.UINT, ctypes.POINTER(ctypes.c_void_p),
                                   W.HANDLE, W.DWORD)
    g.CreateDIBSection.restype = W.HBITMAP
    g.SelectObject.argtypes, g.SelectObject.restype = (W.HDC, W.HGDIOBJ), W.HGDIOBJ
    g.SetStretchBltMode.argtypes = (W.HDC, ctypes.c_int)
    g.SetBrushOrgEx.argtypes = (W.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    g.StretchBlt.argtypes = (W.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             W.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, W.DWORD)
    g.DeleteObject.argtypes = (W.HGDIOBJ,)
    g.DeleteDC.argtypes = (W.HDC,)
    _PROTO_DONE = True


def _win_dpi_aware() -> None:
    if not IS_WIN:
        return
    _win_prototypes()
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _win_process_name(hwnd) -> str:
    from ctypes import wintypes

    pid = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = wintypes.DWORD(512)
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value).lower()
        return ""
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


BROWSERS = ("chrome.exe", "msedge.exe", "firefox.exe", "opera.exe", "brave.exe", "explorer.exe",
            "windowsterminal.exe", "cmd.exe", "conhost.exe", "powershell.exe", "pwsh.exe", "code.exe",
            "discord.exe")


def window_score(title: str, proc: str, want: str) -> int:
    """ウィンドウが目的の Minecraft らしいかの点数（0 は対象外）。"""
    t, w = title.lower(), want.lower()
    if w not in t or "flycraft" in t:
        return 0
    if proc in BROWSERS:
        return 0  # 「Minecraft Wiki - Chrome」などを誤って選ばない
    score = 1
    if t == w:
        score += 3
    if "minecraft" in proc:
        score += 4
    return score


def find_window(title_part: str):
    """Minecraft のウィンドウ (hwnd) を探す（Windows）。見つからなければ None。"""
    if not IS_WIN:
        return None
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found = []
    proc_t = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    own_console = ctypes.windll.kernel32.GetConsoleWindow()

    def cb(hwnd, _):
        if hwnd == own_console:
            return True
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                s = window_score(buf.value, _win_process_name(hwnd), title_part)
                if s >= 2:
                    found.append((s, hwnd, buf.value))
        return True

    user32.EnumWindows(proc_t(cb), 0)
    found.sort(key=lambda t: -t[0])
    return found[0][1] if found else None


def client_rect(hwnd) -> Optional[Rect]:
    if not IS_WIN or not hwnd:
        return None
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    if not user32.IsWindow(hwnd):
        return None
    r = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(r)):
        return None
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, r.right - r.left, r.bottom - r.top


def win_is_foreground(hwnd) -> bool:
    user32 = ctypes.windll.user32
    fg = user32.GetForegroundWindow()
    if fg == hwnd:
        return True
    GA_ROOTOWNER = 3
    return bool(fg) and user32.GetAncestor(fg, GA_ROOTOWNER) == user32.GetAncestor(hwnd, GA_ROOTOWNER)


def win_mouse_acceleration() -> bool:
    """「ポインターの精度を高める」（マウス加速）が有効か。小さな移動ほど縮められて旋回が鈍る。"""
    arr = (ctypes.c_int * 3)()
    if not ctypes.windll.user32.SystemParametersInfoW(0x0003, 0, arr, 0):  # SPI_GETMOUSE
        return False
    return arr[2] != 0


def win_cursor_hidden() -> bool:
    CURSORINFO = win_input_structs()[3]
    ci = CURSORINFO()
    ci.cbSize = ctypes.sizeof(CURSORINFO)
    if not ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci)):
        return False
    return not (ci.flags & 0x1) or not ci.hCursor


class _GdiGrabber:
    """GDI の StretchBlt(HALFTONE) で画面の一部を縮小しながら取り込む（Windows）。"""

    def __init__(self) -> None:
        from ctypes import wintypes

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                        ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

        self.BIH = BITMAPINFOHEADER
        self.user32 = ctypes.windll.user32
        self.gdi = ctypes.windll.gdi32
        _win_prototypes()
        self._size = None
        self._mem = None
        self._bmp = None
        self._bits = None

    def _ensure(self, w: int, h: int) -> None:
        if self._size == (w, h):
            return
        self.close()
        bih = self.BIH()
        bih.biSize = ctypes.sizeof(self.BIH)
        bih.biWidth, bih.biHeight = w, -h  # 上から下
        bih.biPlanes, bih.biBitCount, bih.biCompression = 1, 32, 0
        bits = ctypes.c_void_p()
        screen = self.user32.GetDC(0)
        self._mem = self.gdi.CreateCompatibleDC(screen)
        self._bmp = self.gdi.CreateDIBSection(screen, ctypes.byref(bih), 0, ctypes.byref(bits), None, 0)
        self.user32.ReleaseDC(0, screen)
        self.gdi.SelectObject(self._mem, self._bmp)
        self.gdi.SetStretchBltMode(self._mem, 4)  # HALFTONE（面積平均）
        self.gdi.SetBrushOrgEx(self._mem, 0, 0, None)
        self._bits = bits
        self._size = (w, h)

    def grab(self, rect: Rect, out_w: int, out_h: int) -> np.ndarray:
        self._ensure(out_w, out_h)
        x, y, w, h = rect
        screen = self.user32.GetDC(0)
        try:
            self.gdi.StretchBlt(self._mem, 0, 0, out_w, out_h, screen, x, y, w, h, 0x00CC0020)  # SRCCOPY
        finally:
            self.user32.ReleaseDC(0, screen)
        self.gdi.GdiFlush()  # 描画の完了を待ってからビットを読む
        buf = (ctypes.c_ubyte * (out_w * out_h * 4)).from_address(self._bits.value)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 4)
        return img[:, :, 2::-1].copy()  # BGRA → RGB

    def close(self) -> None:
        if self._bmp:
            self.gdi.DeleteObject(self._bmp)
        if self._mem:
            self.gdi.DeleteDC(self._mem)
        self._bmp = self._mem = None
        self._size = None


# ================================================================= capture
class ScreenCapture:
    """Minecraft のウィンドウ（または指定範囲）を別スレッドで取り込み続ける。"""

    def __init__(self, window: Optional[str] = "Minecraft", region: Optional[str] = None,
                 out_width: int = 256, fps: float = 30.0) -> None:
        _win_dpi_aware()
        self.window = window
        self.region = tuple(int(v) for v in region.split(",")) if region else None
        self.out_width = out_width
        self.fps = fps
        self.hwnd = None
        self._rect: Optional[Rect] = None
        self._rect_t = 0.0
        self._latest: Optional[np.ndarray] = None
        self._latest_t = 0.0
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self.error: Optional[str] = None
        self.method = "gdi"
        if not IS_WIN:
            raise RuntimeError("画面モードは Windows 専用です")

    # ---------------------------------------------------------- location
    def locate(self) -> Rect:
        if self.region:
            return self.region  # type: ignore[return-value]
        now = time.time()
        if self._rect is not None and now - self._rect_t < 1.0:
            return self._rect
        rect = None
        if self.window:
            if not self.hwnd or not ctypes.windll.user32.IsWindow(self.hwnd):
                self.hwnd = find_window(self.window)
            rect = client_rect(self.hwnd) if self.hwnd else None
            if rect and (rect[2] < 50 or rect[3] < 50):
                rect = None
        if rect is None:
            rect = self._monitor()
        self._rect, self._rect_t = rect, now
        return rect

    def _monitor(self) -> Rect:
        u = ctypes.windll.user32
        return 0, 0, u.GetSystemMetrics(0), u.GetSystemMetrics(1)

    # ----------------------------------------------------------- grabbing
    def _grab_once(self, grabber) -> np.ndarray:
        x, y, w, h = self.locate()
        ow = min(self.out_width, w)
        oh = max(1, int(round(h * ow / w)))
        return grabber.grab((x, y, w, h), ow, oh)

    def grab(self) -> np.ndarray:
        """同期的に 1 枚取り込む（スレッド未使用時・較正用）。"""
        if self._thread is not None:
            return self.wait_frame(after=time.time())
        g = _GdiGrabber()
        try:
            return self._grab_once(g)
        finally:
            g.close()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="flycraft-capture", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        g = _GdiGrabber()
        period = 1.0 / self.fps
        try:
            while not self._stop:
                t0 = time.time()
                try:
                    img = self._grab_once(g)
                    self.error = None
                except Exception as e:  # pragma: no cover - 画面の変化など
                    self.error = f"{type(e).__name__}: {e}"
                    time.sleep(0.2)
                    continue
                with self._cond:
                    self._latest, self._latest_t = img, t0
                    self._cond.notify_all()
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            g.close()

    def latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._cond:
            return self._latest, self._latest_t

    def wait_frame(self, after: float, timeout: float = 2.0) -> np.ndarray:
        """時刻 after 以降に撮られた画像を待つ。"""
        end = time.time() + timeout
        with self._cond:
            while self._latest is None or self._latest_t < after:
                rest = end - time.time()
                if rest <= 0:
                    break
                self._cond.wait(rest)
            if self._latest is None:
                raise RuntimeError(self.error or "画面を取り込めません")
            return self._latest

    def close(self) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)

    # -------------------------------------------------------------- state
    def foreground(self) -> bool:
        if self.hwnd:
            return win_is_foreground(self.hwnd)
        # ウィンドウが見つからないのに入力すると、ターミナルやブラウザにキーを送ってしまう。
        # 範囲を明示指定したときだけ（カーソル判定を頼りに）許可する。
        return self.region is not None


# =================================================================== input
def win_input_structs():
    """SendInput / GetCursorInfo 用の構造体。

    Windows のサイズ（DWORD = 32 ビット）に合わせて固定幅の型で定義する
    （他の OS でもサイズの検証ができる）。64 ビットでは INPUT が 40 バイト。
    """
    U32, I32, U16 = ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint16
    ULONG_PTR = ctypes.c_size_t

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", U16), ("wScan", U16), ("dwFlags", U32), ("time", U32), ("dwExtraInfo", ULONG_PTR)]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", I32), ("dy", I32), ("mouseData", U32), ("dwFlags", U32), ("time", U32),
                    ("dwExtraInfo", ULONG_PTR)]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", U32), ("wParamL", U16), ("wParamH", U16)]

    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", U32), ("u", _U)]

    class POINT(ctypes.Structure):
        _fields_ = [("x", I32), ("y", I32)]

    class CURSORINFO(ctypes.Structure):
        _fields_ = [("cbSize", U32), ("flags", U32), ("hCursor", ctypes.c_void_p), ("ptScreenPos", POINT)]

    return INPUT, KEYBDINPUT, MOUSEINPUT, CURSORINFO


class _WinInput:
    SCAN = {"w": 0x11, "a": 0x1E, "s": 0x1F, "d": 0x20, "space": 0x39, "shift": 0x2A}

    def __init__(self) -> None:
        from ctypes import wintypes

        INPUT, KEYBDINPUT, MOUSEINPUT, _ = win_input_structs()
        self.INPUT, self.KEYBDINPUT, self.MOUSEINPUT = INPUT, KEYBDINPUT, MOUSEINPUT
        self.user32 = ctypes.windll.user32
        self.user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self._prev: Dict[int, bool] = {}

    def _send(self, inp) -> None:
        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def key(self, name: str, down: bool) -> None:
        flags = 0x0008 | (0 if down else 0x0002)  # SCANCODE | KEYUP
        inp = self.INPUT(type=1)
        inp.u.ki = self.KEYBDINPUT(0, self.SCAN[name], flags, 0, 0)
        self._send(inp)

    def move(self, dx: int, dy: int) -> None:
        inp = self.INPUT(type=0)
        inp.u.mi = self.MOUSEINPUT(int(dx), int(dy), 0, 0x0001, 0, 0)  # MOUSEEVENTF_MOVE（相対）
        self._send(inp)

    def button(self, down: bool) -> None:
        inp = self.INPUT(type=0)
        inp.u.mi = self.MOUSEINPUT(0, 0, 0, 0x0002 if down else 0x0004, 0, 0)
        self._send(inp)

    def hotkey_pressed(self, vk: int) -> bool:
        """押された瞬間だけ True（上位ビットの立ち上がりを自前で検出）。"""
        now = bool(self.user32.GetAsyncKeyState(vk) & 0x8000)
        was = self._prev.get(vk, False)
        self._prev[vk] = now
        return now and not was


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

    def hotkey_pressed(self, vk):
        return False


def make_input():
    if not IS_WIN:
        raise RuntimeError("画面モードは Windows 専用です")
    return _WinInput()


# ============================================================== controller
@dataclass
class KeyPolicy:
    fwd_on: float = 0.3  # これを超えたら W を押す
    fwd_off: float = 0.12  # これを下回ったら離す（ヒステリシス）
    min_hold: float = 0.25  # 押したら最低この時間は押し続ける [s]
    min_gap: float = 0.45  # 離してから再び押すまでの最小間隔（2 度押しダッシュ防止）[s]
    jump_hold: float = 0.12  # Space を押している時間 [s]
    jump_cooldown: float = 0.6  # ジャンプの間隔（2 度押しで飛行モードにならないように）[s]
    attack_release: float = 0.3  # 噛む指令が消えてから左クリックを離すまで [s]
    turn_deg_s: float = 360.0  # 旋回指令 1.0 のときの回転速度 [度/秒]（ハエの急旋回は 500°/秒を超える）
    max_turn_step: float = 60.0  # 1 回で回す角度の上限 [度]


class _Key:
    def __init__(self, send, name: str) -> None:
        self.send, self.name = send, name
        self.down = False
        self.t_change = -1e9

    def set(self, on: bool, now: float) -> None:
        if on != self.down:
            self.down = on
            self.t_change = now
            self.send(self.name, on)


class Controller:
    """脳の指令 → キー・マウス操作（統合版の癖を避けるための時間的な制約つき）。"""

    def __init__(self, dev, policy: Optional[KeyPolicy] = None, deg_per_px: float = 0.15) -> None:
        self.dev = dev
        self.p = policy or KeyPolicy()
        self.deg_per_px = deg_per_px
        send = lambda name, on: self.dev.button(on) if name == "mouse" else self.dev.key(name, on)  # noqa: E731
        self.keys = {k: _Key(send, k) for k in ("w", "s", "space", "mouse")}
        self._last_jump = -1e9
        self._last_attack = -1e9
        self._mx = 0.0
        self.turned_deg = 0.0
        self.px_total = 0  # 送ったマウス移動量の合計 [px]

    def _hold(self, key: str, want: bool, now: float) -> None:
        k, p = self.keys[key], self.p
        if want and not k.down:
            if now - k.t_change >= p.min_gap:
                k.set(True, now)
        elif not want and k.down:
            if now - k.t_change >= p.min_hold:
                k.set(False, now)

    def apply(self, action: Action, now: float, wall_dt: float) -> None:
        p = self.p
        w, s = self.keys["w"], self.keys["s"]
        f = float(action.forward)
        want_w = f > p.fwd_on or (w.down and f > p.fwd_off)
        want_s = f < -p.fwd_on or (s.down and f < -p.fwd_off)
        if want_w and s.down:
            want_s = False
        if want_s and w.down:
            want_w = False
        self._hold("w", want_w, now)
        self._hold("s", want_s, now)
        # ジャンプ: 短く押して離す。クールダウン中は押さない
        sp = self.keys["space"]
        if sp.down and now - sp.t_change >= p.jump_hold:
            sp.set(False, now)
        elif not sp.down and action.jump and now - self._last_jump >= p.jump_cooldown:
            sp.set(True, now)
            self._last_jump = now
        # 噛む: 指令が消えてもしばらく押し続ける（採掘が途切れないように）
        if action.attack:
            self._last_attack = now
        want_m = now - self._last_attack < p.attack_release
        m = self.keys["mouse"]
        if want_m != m.down and (want_m or now - m.t_change >= p.min_hold):
            m.set(want_m, now)
        # 旋回: 実時間に比例した角度 → マウスの移動量（+ = 左 → マウスは左 = dx < 0）
        deg = float(np.clip(action.turn * p.turn_deg_s * wall_dt, -p.max_turn_step, p.max_turn_step))
        self._mx += -deg / max(self.deg_per_px, 1e-4)
        step = int(self._mx)
        if step:
            self._mx -= step
            self.dev.move(step, 0)
            self.px_total += abs(step)
            self.turned_deg += -step * self.deg_per_px

    def release_all(self, now: float) -> None:
        for k in self.keys.values():
            k.set(False, now)
        self._mx = 0.0

    def pressed(self) -> Dict[str, bool]:
        return {k: v.down for k, v in self.keys.items()}


# =========================================================== image motion
def _gray_small(frame: np.ndarray, w: int = 96) -> np.ndarray:
    g = frame.astype(np.float32).mean(axis=2)
    k = max(1, g.shape[1] // w)
    h2, w2 = (g.shape[0] // k) * k, (g.shape[1] // k) * k
    return g[:h2, :w2].reshape(h2 // k, k, w2 // k, k).mean(axis=(1, 3))


def phase_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """b が a に対して (dx, dy) だけずれている量（位相限定相関）と、ピークの鋭さ。"""
    wy = np.hanning(a.shape[0])[:, None]
    wx = np.hanning(a.shape[1])[None, :]
    fa = np.fft.fft2((a - a.mean()) * wy * wx)
    fb = np.fft.fft2((b - b.mean()) * wy * wx)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-9
    c = np.real(np.fft.ifft2(r))
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    peak = float(c[iy, ix])
    dy = iy if iy <= a.shape[0] // 2 else iy - a.shape[0]
    dx = ix if ix <= a.shape[1] // 2 else ix - a.shape[1]
    # 放物線近似でサブピクセル
    def sub(cm, c0, cp):
        d = cm - 2 * c0 + cp
        return 0.0 if abs(d) < 1e-12 else 0.5 * (cm - cp) / d

    fx = sub(c[iy, (ix - 1) % c.shape[1]], c[iy, ix], c[iy, (ix + 1) % c.shape[1]])
    fy = sub(c[(iy - 1) % c.shape[0], ix], c[iy, ix], c[(iy + 1) % c.shape[0], ix])
    return dx + fx, dy + fy, peak


def parallax_residual(a: np.ndarray, b: np.ndarray, fov_h: float = 100.0, r: int = 1) -> float:
    """回転（と視点の揺れ）を補正したあとに残る画面の変化の大きさ。

    前に進んでいれば近い物ほど大きく動く（視差）ので残差が大きく、壁に押し付けられて
    止まっていれば小さい。画面中央のずれから回転角を推定し、透視投影に沿って前の画像を
    回転させてから比べる。値はコントラストで正規化する。
    """
    H, W = a.shape
    f = (W / 2) / math.tan(math.radians(fov_h) / 2)
    ch, cw = H // 4, W // 4
    dx, dy, _ = phase_shift(a[ch:H - ch, cw:W - cw], b[ch:H - ch, cw:W - cw])
    delta = math.atan(dx / f)
    xs = np.arange(W) - (W - 1) / 2
    ys = np.arange(H) - (H - 1) / 2
    phi_b = np.arctan(xs / f)
    phi_a = phi_b - delta
    ok_x = np.abs(phi_a) < math.radians(fov_h) / 2
    xi = np.rint((W - 1) / 2 + f * np.tan(np.clip(phi_a, -1.5, 1.5))).astype(int)
    scale = np.cos(phi_b) / np.cos(phi_a)
    best = np.inf
    for sy in range(int(round(dy)) - r, int(round(dy)) + r + 1):
        yi = np.rint((H - 1) / 2 + (ys[:, None] - sy) * scale[None, :]).astype(int)
        xx = np.broadcast_to(xi, (H, W))
        m = ok_x[None, :] & (xx >= 0) & (xx < W) & (yi >= 0) & (yi < H)
        if m.mean() < 0.3:
            continue
        best = min(best, float(np.abs(b[m] - a[yi[m], xx[m]]).mean()))
    if not np.isfinite(best):
        return 1.0
    contrast = float(np.abs(a - a.mean()).mean()) + 1.0
    return best / contrast


class BumpDetector:
    """W を押しているのに足元の地面が流れない状態が続いたら「何かに当たっている」とみなす。

    歩いていれば足元の地面は近いので大きく・不均一に流れる（視差）。壁に押し付けられて
    いれば、視点の揺れや旋回で画面がずれても、回転を補正した残差は小さい。
    画面下部の左寄り（ホットバーと右下の手を避ける）で判定し、旋回が大きいほど閾値を上げる。
    FakeCraft（統合版の操作系を真似たテスト用ゲーム）での実測: 押し付け 0.03〜0.05、
    その場で旋回しながら穴の中 0.2〜0.4、歩行 0.6〜1.2。
    """

    REGION = (0.55, 0.88, 0.08, 0.62)  # 行・列の範囲（割合）

    def __init__(self, interval: float = 0.12, threshold: float = 0.22, per_deg: float = 0.03,
                 hold: float = 0.35, max_turn_deg: float = 10.0) -> None:
        self.interval = interval
        self.threshold = threshold
        self.per_deg = per_deg
        self.hold = hold
        self.max_turn_deg = max_turn_deg
        self._ref = None
        self._ref_t = 0.0
        self._ref_turn = 0.0
        self._stuck_t = 0.0
        self.residual = 0.0
        self.touch = 0.0

    def update(self, frame: np.ndarray, t: float, walking: bool, dt: float, turned_deg: float = 0.0,
               fov_h: float = 100.0) -> float:
        g = _gray_small(frame)
        H, W = g.shape
        y0, y1, x0, x1 = self.REGION
        g = g[int(H * y0):int(H * y1), int(W * x0):int(W * x1)]
        if self._ref is None or self._ref.shape != g.shape:
            self._ref, self._ref_t, self._ref_turn = g, t, turned_deg
        elif t - self._ref_t >= self.interval:
            span = t - self._ref_t
            turned = abs(turned_deg - self._ref_turn) * (self.interval / span)
            if turned <= self.max_turn_deg:
                self.residual = parallax_residual(self._ref, g, fov_h * (x1 - x0))
                if walking and self.residual < self.threshold + self.per_deg * turned:
                    self._stuck_t += span
                else:
                    self._stuck_t = 0.0
            self._ref, self._ref_t, self._ref_turn = g, t, turned_deg
        if not walking:
            self._stuck_t = 0.0
        self.touch = 0.8 if self._stuck_t >= self.hold else self.touch * math.exp(-dt / 0.3)
        return self.touch


class ViewCheck:
    """マウスを動かしたら視点が回るか（= ゲーム画面か）を画像で確かめる。

    統合版ではメニューやチャットでもカーソルの状態で判定できない場合があるため、
    「送ったマウス移動に対して画面が回転したか」を直接見る。ハエがあまり旋回しないときは
    数秒ごとに小さく（約 3°）左右に振って確かめる。回らなければ入力を止め、
    回るようになったら再開する。
    """

    def __init__(self, probe_deg: float = 3.0, probe_every: float = 2.5, lag: float = 0.08) -> None:
        self.probe_deg = probe_deg
        self.probe_every = probe_every
        self.lag = lag
        self.total = 0.0  # 送った回転の累計 [度]（+ = 左）
        self.hist = [(0.0, 0.0)]  # (時刻, 累計)
        self.frames = []  # (時刻, 画面中央の灰色画像, 倍率)
        self.ok = True
        self.fails = 0
        self.last_eval = 0.0
        self.last_checked = time.time()
        self.probe_back = None  # (戻す時刻, px)
        self.observed = 0.0
        self.expected = 0.0

    def record_move(self, t: float, deg: float) -> None:
        if deg:
            self.total += deg
            self.hist.append((t, self.total))
            self.hist = self.hist[-400:]

    def _turned_at(self, t: float) -> float:
        v = self.hist[0][1]
        for tt, tot in self.hist:
            if tt <= t:
                v = tot
            else:
                break
        return v

    def add_frame(self, t: float, frame: np.ndarray, focal: float) -> None:
        if self.frames and self.frames[-1][0] == t:
            return
        g = _gray_small(frame, w=192)
        h, w = g.shape
        crop = g[int(h * 0.2):int(h * 0.8), int(w * 0.25):int(w * 0.75)]
        self.frames.append((t, crop, frame.shape[1] / w / focal))
        self.frames = [f for f in self.frames if f[0] > t - 1.5]
        self._evaluate(t)

    def _evaluate(self, t1: float) -> None:
        if t1 - self.last_eval < 0.25 or len(self.frames) < 3:
            return
        f1 = self.frames[-1]
        cands = [f for f in self.frames if 0.25 <= t1 - f[0] <= 0.7]
        if not cands:
            return
        f0 = cands[-1]
        exp = self._turned_at(t1 - self.lag) - self._turned_at(f0[0] - self.lag)
        if not 2.5 <= abs(exp) <= 20.0:
            return
        self.last_eval = t1
        dx, _, peak = phase_shift(f0[1], f1[1])
        obs = math.degrees(math.atan(dx * f1[2]))  # 左を向くと画面は右へずれる（dx > 0）
        self.observed, self.expected = obs, exp
        self.last_checked = t1
        if peak >= 0.05 and obs / exp > 0.3:
            self.fails = 0
            self.ok = True
        else:
            self.fails += 1
            if self.fails >= 2:
                self.ok = False

    def probe(self, t: float, dev, deg_per_px: float) -> None:
        """必要なら確認のために小さく振る（行って、少しあとで戻す）。"""
        if self.probe_back is not None:
            t_back, px = self.probe_back
            if t >= t_back:
                dev.move(-px, 0)
                self.record_move(t, px * deg_per_px)
                self.probe_back = None
            return
        wait = self.probe_every if self.ok else 1.0
        if t - self.last_checked >= wait:
            px = max(1, int(round(self.probe_deg / max(deg_per_px, 1e-4))))
            dev.move(px, 0)  # 右へ
            self.record_move(t, -px * deg_per_px)
            self.probe_back = (t + 0.45, px)
            self.last_checked = t


# ================================================================ backend
class ScreenBackend(Backend):
    name = "screen"
    VK_F8 = 0x77

    def __init__(self, window: Optional[str] = "Minecraft", region: Optional[str] = None,
                 fov_v: float = 70.0, deg_per_px: Optional[float] = None, calibrate: bool = True,
                 pitch: Optional[float] = 8.0, cursor_check: bool = True, countdown: float = 3.0,
                 policy: Optional[KeyPolicy] = None, capture=None, input_device=None, check_view: bool = True,
                 log: Callable[[str], None] = print) -> None:
        self.cap = capture or ScreenCapture(window, region)
        self.dev = input_device or make_input()
        self.ctl = Controller(self.dev, policy, deg_per_px or 0.15)
        self.fov_v = fov_v
        self.calibrate = calibrate and deg_per_px is None
        self.target_pitch = pitch
        self.cursor_check = cursor_check
        self.check_view = check_view
        self.countdown = countdown
        self.log = log
        self.paused = False
        self.bump = BumpDetector()
        self.view = ViewCheck()
        self.status = ""
        self.events = []
        self._t_last = time.perf_counter()
        self._t_step = 0.0
        self._walk_since = None
        self._was_active = None
        self.turn_rate = 0.0  # 実際に回している速さ [度/秒]
        self.px_rate = 0.0  # 送っているマウス移動量 [px/秒]
        atexit.register(self._emergency_release)

    @property
    def realtime(self) -> bool:
        return True

    # --------------------------------------------------------------- state
    def cursor_hidden(self) -> Optional[bool]:
        if not self.cursor_check:
            return True
        if IS_WIN:
            return win_cursor_hidden()
        return None

    def game_active(self) -> Tuple[bool, str]:
        if self.paused:
            return False, "F8 で一時停止中"
        if not self.cap.foreground():
            if not self.cap.hwnd and not self.cap.region:
                return False, "Minecraft のウィンドウが見つかりません（起動しているか、--window を確認）"
            return False, "Minecraft が最前面ではありません（クリックして前面に）"
        hidden = self.cursor_hidden()
        if hidden is False:
            return False, "マウスカーソルが表示されています（メニュー・チャット中？ Esc でゲームに戻す）"
        if not self.view.ok:
            return False, "マウスで視点が回りません（メニュー・チャット・インベントリ中？ Esc でゲームに戻す）"
        return True, "操作中"

    # --------------------------------------------------------------- setup
    def reset(self) -> Observation:
        self.cap.start()
        frame = self.cap.wait_frame(after=0.0, timeout=5.0)
        x, y, w, h = self.cap.locate()
        how = "ウィンドウ" if self.cap.hwnd else ("指定範囲" if self.cap.region else "画面全体")
        self.log(f"🎮 キャプチャ: {how} x={x} y={y} {w}×{h} → {frame.shape[1]}×{frame.shape[0]}（{self.cap.method}）")
        if not self.cap.hwnd and not self.cap.region:
            self.log("   ⚠ Minecraft のウィンドウが見つかりません。起動しているか、--window でタイトルを指定してください")
        self.log("   ・Minecraft を最前面にしてゲーム画面（メニューが閉じた状態）にすると操作が始まります")
        self.log("   ・F8 で一時停止/再開、ターミナルで Ctrl+C で終了")
        if IS_WIN:
            try:
                if win_mouse_acceleration():
                    self.log("   ⚠ Windows の「ポインターの精度を高める」が有効です。小さなマウス移動が縮められて"
                             "旋回が鈍くなることがあります（設定 > Bluetooth とデバイス > マウス > マウスの追加設定 > "
                             "ポインター オプション でオフにするのがおすすめ）")
            except Exception:
                pass
        for k in range(int(self.countdown), 0, -1):
            self.log(f"   {k}…")
            time.sleep(1.0)
        if self.calibrate:
            self._calibrate()
        return self._observe(0.05)

    def _wait_active(self, timeout: float = 30.0) -> bool:
        end = time.time() + timeout
        shown = False
        while time.time() < end:
            ok, why = self.game_active()
            if ok:
                return True
            if not shown:
                self.log(f"   …較正の待機中: {why}")
                shown = True
            time.sleep(0.2)
        return False

    def _measure_dpp(self) -> Optional[float]:
        """マウスを左右に動かし、画面中央部のずれから 1 px あたりの回転角を測る。"""
        frame = self.cap.wait_frame(after=time.time())
        H, W = frame.shape[:2]
        fov_h = math.degrees(2 * math.atan(math.tan(math.radians(self.fov_v) / 2) * W / H))
        focal = (W / 2) / math.tan(math.radians(fov_h) / 2)

        def center(img):
            g = _gray_small(img, w=192)
            h, w = g.shape
            return g[int(h * 0.2):int(h * 0.8), int(w * 0.25):int(w * 0.75)], W / g.shape[1]

        def turn_and_measure(n: int, sign: int):
            """マウスを横に n px 動かし、画面の回転角 [度] を測る（測れなければ None）。"""
            a, k = center(self.cap.wait_frame(after=time.time() + 0.05))
            chunk = max(1, n // 6)  # 実際のプレイと同じく小刻みに動かす
            left = n
            while left > 0:
                step = min(chunk, left)
                self.dev.move(sign * step, 0)
                left -= step
                time.sleep(0.02)
            b, _ = center(self.cap.wait_frame(after=time.time() + 0.25))
            dx, dy, peak = phase_shift(a, b)
            ang = math.degrees(math.atan(abs(dx) * k / focal))
            if peak < 0.05 or (dx < 0) != (sign > 0):  # 右を向けば画面は左へずれる
                return None
            return ang

        # 1) 小さく動かして大まかな感度をつかむ（感度が高くても低くても測れるように）
        guess = None
        for n in (12, 48, 192, 768, 3072):
            ang = turn_and_measure(n, 1)
            turn_and_measure(n, -1)
            if ang is not None and ang >= 1.5:
                guess = ang / n
                break
        if guess is None:
            return None
        # 2) 約 8° 回る量で左右 2 往復して精密に測る
        n = int(np.clip(round(8.0 / guess), 4, 20000))
        results = []
        for sign in (1, -1, 1, -1):
            ang = turn_and_measure(n, sign)
            if ang is not None and ang > 0.3:
                results.append(ang / n)
        if len(results) >= 2:
            dpp = float(np.median(results))
            if 0.0005 < dpp < 10.0:
                return dpp
        return None

    def _level_pitch(self) -> None:
        """真下を向いてから決まった角度だけ上げる（ピッチは ±90° で止まる）。"""
        dpp = self.ctl.deg_per_px
        down = int(120 / dpp)
        for _ in range(4):
            self.dev.move(0, down // 4)
            time.sleep(0.03)
        time.sleep(0.15)
        up = int(round((90 - self.target_pitch) / dpp))
        for k in range(4):
            self.dev.move(0, -(up // 4 + (up % 4 if k == 3 else 0)))
            time.sleep(0.03)
        time.sleep(0.15)

    def _calibrate(self) -> None:
        """マウス 1 px あたりの回転角を測り、視線を水平付近にそろえる（2 回繰り返して精度を上げる）。"""
        if not self._wait_active():
            self.log("   ⚠ 較正をスキップ（ゲーム画面になりませんでした）。--deg-per-px で指定もできます")
            return
        ok = False
        for rnd in range(2):
            dpp = self._measure_dpp()
            if dpp is not None:
                self.ctl.deg_per_px = dpp
                ok = True
            if self.target_pitch is not None and self._wait_active(5.0):
                self._level_pitch()
        if ok:
            self.log(f"🖱  マウス較正: 1 px ≈ {self.ctl.deg_per_px:.3f}°（旋回 1.0 = {self.ctl.p.turn_deg_s:.0f}°/秒）")
        else:
            self.log(f"   ⚠ マウス較正に失敗しました。1 px = {self.ctl.deg_per_px:.3f}° として続けます（--deg-per-px で指定可）")
        if self.target_pitch is not None:
            self.log(f"   視線を水平から {self.target_pitch:.0f}° 下にそろえました")

    # ---------------------------------------------------------------- loop
    def _observe(self, dt: float) -> Observation:
        frame, t = self.cap.latest()
        if frame is None:
            frame = self.cap.wait_frame(after=0.0)
            t = time.time()
        walking = self.ctl.keys["w"].down and self._walk_since is not None and t - self._walk_since > 0.3
        fov_h = math.degrees(2 * math.atan(math.tan(math.radians(self.fov_v) / 2) * frame.shape[1] / frame.shape[0]))
        touch = self.bump.update(frame, t, walking, dt, self.ctl.turned_deg, fov_h)
        if self.check_view:
            focal = (frame.shape[1] / 2) / math.tan(math.radians(fov_h) / 2)
            self.view.add_frame(t, frame, focal)
        return Observation(frame=frame, fov_v=self.fov_v, touch_left=touch, touch_right=touch,
                           info={"状態": self.status, "キー": "".join(k.upper() for k, v in self.ctl.pressed().items() if v),
                                 "視差": round(self.bump.residual, 3)})

    def step(self, action: Action, dt: float) -> Observation:
        now = time.perf_counter()
        # 前回の操作から今回までの実時間（待ち時間も含める。以前は待ち時間を除いてしまい旋回が遅かった）
        wall_dt = min(0.2, max(0.0, now - self._t_step)) if self._t_step else 0.05
        self._t_step = now
        if self.dev.hotkey_pressed(self.VK_F8):
            self.paused = not self.paused
            self.log("⏸ 一時停止（F8）" if self.paused else "▶ 再開（F8）")
        active, why = self.game_active()
        # メニュー等で視点が回らなくなっていないか確かめる（前面・一時停止でない・カーソル非表示のときだけ）
        if active or (why.startswith("マウスで視点") and self.check_view):
            self.view.probe(time.time(), self.dev, self.ctl.deg_per_px)
            active, why = self.game_active()
        self.status = why
        if active != self._was_active:
            self._was_active = active
            if not active:
                self.log(f"   ⏹ 入力停止: {why}")
            else:
                self.log("   ▶ 操作を再開")
            self.events.append(why)
            self.events = self.events[-20:]
        mono = time.monotonic()
        before = self.ctl.turned_deg
        px_before = self.ctl.px_total
        if active:
            self.ctl.apply(action, mono, wall_dt)
        else:
            self.ctl.release_all(mono)
        self.view.record_move(time.time(), self.ctl.turned_deg - before)
        if wall_dt > 0:
            inst = abs(self.ctl.turned_deg - before) / wall_dt
            k = min(1.0, wall_dt / 1.0)  # 約 1 秒の平均
            self.turn_rate += k * (inst - self.turn_rate)
            self.px_rate += k * ((self.ctl.px_total - px_before) / wall_dt - self.px_rate)
        if self.ctl.keys["w"].down:
            if self._walk_since is None:
                self._walk_since = time.time()
        else:
            self._walk_since = None
        # 実時間に合わせる（脳のほうが速いとき）
        wait = dt - (time.perf_counter() - self._t_last)
        if wait > 0:
            time.sleep(wait)
        self._t_last = time.perf_counter()
        return self._observe(dt)

    def set_turn_speed(self, deg_s: float) -> None:
        self.ctl.p.turn_deg_s = float(np.clip(deg_s, 30, 1440))

    def extra_telemetry(self):
        return {"status": self.status, "events": list(self.events[-8:]),
                "keys": self.ctl.pressed(), "deg_per_px": round(self.ctl.deg_per_px, 4),
                "turn_deg_s": round(self.ctl.p.turn_deg_s), "turn_rate": round(self.turn_rate),
                "mouse_px_s": round(self.px_rate),
                "view_check": {"ok": self.view.ok, "expected": round(self.view.expected, 1),
                               "observed": round(self.view.observed, 1)}}

    def _emergency_release(self) -> None:
        try:
            self.ctl.release_all(time.monotonic())
        except Exception:
            pass

    def close(self) -> None:
        self._emergency_release()
        try:
            self.cap.close()
        except Exception:
            pass

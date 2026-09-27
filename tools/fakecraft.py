"""FakeCraft: 画面モード（方法 B）のテストプレイ用に、Minecraft 統合版 PC 版の
操作系を真似たゲームウィンドウ。

FlyCraft の `screen` モードは本物のゲームと同じように、このウィンドウを画面キャプチャし、
仮想キーボード・マウス（SendInput）で操作する。統合版の次の癖を再現して、
ハエの操作が意図しない動作を起こさないかを記録する:

* W を素早く 2 回押す → ダッシュ（sprint）
* クリエイティブで Space を素早く 2 回押す → 飛行モードの切り替え
* フォーカスを失う / Esc → ポーズメニュー（マウスカーソルが出る。ボタンをクリックできる）
* 視点の揺れ（View Bobbing）、手の表示、HUD（照準・ホットバー・体力）
* 自動ジャンプ、左クリック長押しで採掘

使い方（Windows、ターミナルを 2 つ）:
    python tools/fakecraft.py --log play.json --pause-every 30
    python -m flycraft screen
FakeCraft のウィンドウ名は「Minecraft」なので、そのまま画面モードの対象になる
（本物の Minecraft は閉じておく）。終了後の play.json に、ダッシュ・飛行の誤発動や
ポーズ中の誤クリック、詰まっていた秒数などが記録される。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame  # noqa: E402

from flycraft.backends import sim  # noqa: E402
from flycraft.backends.sim import SimWorld  # noqa: E402
from flycraft.interface import Action  # noqa: E402

DOUBLE_TAP = 0.3  # 統合版の 2 度押し判定 [s]（おおよそ）
DEG_PER_PX = 0.15  # マウス感度（既定の感度に近い値）


class FakeCraft:
    def __init__(self, args):
        self.args = args
        pygame.init()
        self.W, self.H = args.width, args.height
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Minecraft")
        self.world = SimWorld(seed=args.seed, width=args.render_w, height=int(args.render_w * self.H / self.W),
                              fov_v=args.fov, n_slimes=args.slimes)
        self.world.reset()
        self.world.player.pitch = -5.0
        self.font = pygame.font.Font(None, 22)
        self.keys = {"w": False, "s": False, "a": False, "d": False, "space": False}
        self.mouse_down = False
        self.jump_latch = False
        self.frames = 0
        self.paused = False
        self.creative = args.creative
        self.flying = False
        self.sprint = False
        self.last_w_release = -9.0
        self.last_w_press = -9.0
        self.last_space_press = -9.0
        self.bob_t = 0.0
        self.t0 = time.time()
        self.stats = {"sprints": 0, "fly_toggles": 0, "menu_clicks": 0, "quit_clicks": 0, "clicks_while_paused": 0,
                      "keys_while_paused": 0, "pause_seconds": 0.0, "jump_presses": 0, "w_presses": 0,
                      "mouse_px": 0.0, "yaw_turned": 0.0, "stuck_seconds": 0, "moving_seconds": 0,
                      "w_held_seconds": 0.0}
        self._pos_hist = []
        self.set_paused(False)

    # ------------------------------------------------------------ helpers
    def set_paused(self, on: bool):
        # 統合版と同じく、プレイ中はカーソルを隠してマウスを中央に固定、メニュー中はカーソルを出す
        self.paused = on
        pygame.mouse.set_visible(on)
        if not on:
            pygame.mouse.set_pos(self.W // 2, self.H // 2)
            pygame.event.clear(pygame.MOUSEMOTION)
        if on:
            for k in self.keys:
                self.keys[k] = False
            self.mouse_down = False

    def now(self):
        return time.time() - self.t0

    # -------------------------------------------------------------- input
    def handle(self, ev):
        t = self.now()
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.ACTIVEEVENT and getattr(ev, "gain", 1) == 0 and getattr(ev, "state", 0) & 2:
            self.set_paused(True)
        if ev.type == pygame.WINDOWFOCUSLOST:
            self.set_paused(True)
        if ev.type == pygame.KEYDOWN:
            if self.paused:
                self.stats["keys_while_paused"] += 1
            if ev.key == pygame.K_ESCAPE:
                self.set_paused(not self.paused)
                return True
            name = {pygame.K_w: "w", pygame.K_s: "s", pygame.K_a: "a", pygame.K_d: "d", pygame.K_SPACE: "space"}.get(ev.key)
            if name and not self.paused:
                self.keys[name] = True
                if name == "w":
                    self.stats["w_presses"] += 1
                    if t - self.last_w_release < DOUBLE_TAP and t - self.last_w_press < 2 * DOUBLE_TAP:
                        self.sprint = True
                        self.stats["sprints"] += 1
                    self.last_w_press = t
                if name == "space":
                    self.stats["jump_presses"] += 1
                    self.jump_latch = True  # 統合版と同じく、押した瞬間を取りこぼさない
                    if self.creative and t - self.last_space_press < DOUBLE_TAP:
                        self.flying = not self.flying
                        self.stats["fly_toggles"] += 1
                    self.last_space_press = t
        if ev.type == pygame.KEYUP:
            name = {pygame.K_w: "w", pygame.K_s: "s", pygame.K_a: "a", pygame.K_d: "d", pygame.K_SPACE: "space"}.get(ev.key)
            if name:
                self.keys[name] = False
                if name == "w":
                    self.last_w_release = t
                    self.sprint = False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.paused:
                self.stats["clicks_while_paused"] += 1
                x, y = ev.pos
                if self._btn_resume.collidepoint(x, y):
                    self.stats["menu_clicks"] += 1
                    self.set_paused(False)
                elif self._btn_quit.collidepoint(x, y):
                    self.stats["quit_clicks"] += 1
                    print("!!! 『保存してタイトルへ』がクリックされた", flush=True)
            else:
                self.mouse_down = True
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.mouse_down = False
        return True

    def poll_mouse(self):
        """カーソル位置の中央からのずれを視点の回転にして、中央へ戻す。"""
        if self.paused:
            return
        x, y = pygame.mouse.get_pos()
        dx, dy = x - self.W // 2, y - self.H // 2
        if dx or dy:
            p = self.world.player
            p.yaw = (p.yaw - dx * DEG_PER_PX) % 360  # 右へ動かす → 右を向く（シムの yaw は左回りが正）
            p.pitch = float(np.clip(p.pitch - dy * DEG_PER_PX, -89, 89))
            self.stats["mouse_px"] += abs(dx)
            self.stats["yaw_turned"] += dx * DEG_PER_PX
            pygame.mouse.set_pos(self.W // 2, self.H // 2)

    # ---------------------------------------------------------- simulate
    def update(self, dt):
        if self.paused:
            self.stats["pause_seconds"] += dt
            return
        fwd = (1.0 if self.keys["w"] else 0.0) - (1.0 if self.keys["s"] else 0.0)
        sim.WALK_SPEED = 4.3 * (1.3 if self.sprint else 1.0)
        # 統合版と同じく、物理は描画速度と無関係に 20 ティック/秒で進める
        self._acc = getattr(self, "_acc", 0.0) + dt
        while self._acc >= 0.05:
            self._acc -= 0.05
            a = Action(forward=fwd, jump=self.keys["space"] or self.jump_latch, attack=self.mouse_down)
            self.jump_latch = False
            self.world.step(a, 0.05)
        if self.keys["w"]:
            self.stats["w_held_seconds"] += dt
        # 視点の揺れ
        walking = abs(fwd) > 0 and self.world.player.on_ground
        self.bob_t += dt * (2.0 if walking else 0.0)
        amp = 1.0 if (walking and self.args.bobbing) else 0.0
        self.world.view_offset = [0.0, amp * 0.06 * abs(math.sin(math.pi * self.bob_t)), 0.0,
                                  amp * 0.8 * math.sin(math.pi * self.bob_t), amp * 0.6 * abs(math.cos(math.pi * self.bob_t))]
        p = self.world.player
        self._pos_hist.append((self.now(), p.x, p.z, self.keys["w"]))

    def second_stats(self):
        h = self._pos_hist
        if len(h) < 2:
            return
        t_end = h[-1][0]
        old = [e for e in h if e[0] <= t_end - 1.0]
        if not old:
            return
        o = old[-1]
        e = h[-1]
        if any(x[3] for x in h if x[0] > o[0]):
            if math.hypot(e[1] - o[1], e[2] - o[2]) < 0.3:
                self.stats["stuck_seconds"] += 1
                w, p = self.world, self.world.player
                fx, fz = w._forward_vec()
                feet = w._block_at(p.x + fx * 0.7, p.y + 0.5, p.z + fz * 0.7)
                head = w._block_at(p.x + fx * 0.7, p.y + 1.5, p.z + fz * 0.7)
                above = w._block_at(p.x, p.y + 2.2, p.z)
                kind = ("step" if feet and not head else "wall" if feet and head else
                        "head" if head else "open")
                if above:
                    kind += "+ceiling"
                if not p.on_ground:
                    kind += "+air"
                from flycraft.backends.sim import WATER
                if w._block_at(p.x, p.y + 0.5, p.z) == WATER:
                    kind += "+water"
                ring = sum(1 for ddx in (-1, 0, 1) for ddz in (-1, 0, 1) if (ddx or ddz)
                           and w._block_at(p.x + ddx * 0.8, p.y + 0.5, p.z + ddz * 0.8))
                self.stats.setdefault("stuck_log", []).append(
                    [round(p.x, 1), round(p.y, 1), round(p.z, 1), round(p.yaw), round(p.vy, 1), kind, ring])
                self.stats["stuck_log"] = self.stats["stuck_log"][-40:]
                self.stats.setdefault("stuck_kind", {})
                self.stats["stuck_kind"][kind] = self.stats["stuck_kind"].get(kind, 0) + 1
            else:
                self.stats["moving_seconds"] += 1
        self._pos_hist = [x for x in h if x[0] > t_end - 1.0]

    # ------------------------------------------------------------- draw
    def draw(self):
        frame = self.world.render()
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (self.W, self.H))
        self.screen.blit(surf, (0, 0))
        W, H = self.W, self.H
        if self.args.hud:
            # 手（揺れる）
            if self.args.hand:
                bx = int(W * 0.78 + 12 * math.sin(math.pi * self.bob_t))
                by = int(H * 0.72 + 10 * abs(math.cos(math.pi * self.bob_t)))
                if self.mouse_down:
                    by -= int(20 * abs(math.sin(self.now() * 12)))
                pygame.draw.rect(self.screen, (196, 140, 100), (bx, by, 70, 160))
                pygame.draw.rect(self.screen, (160, 110, 80), (bx, by, 70, 160), 3)
            # ホットバー
            hw = int(W * 0.45)
            x0 = (W - hw) // 2
            for i in range(9):
                r = pygame.Rect(x0 + i * hw // 9, H - 44, hw // 9 - 2, 40)
                pygame.draw.rect(self.screen, (60, 60, 60), r)
                pygame.draw.rect(self.screen, (200, 200, 200) if i == 0 else (110, 110, 110), r, 2)
            # 体力
            hp = int(self.world.player.health)
            for i in range(10):
                c = (220, 30, 30) if hp >= (i + 1) * 2 else (70, 20, 20)
                pygame.draw.rect(self.screen, c, (x0 + i * 16, H - 64, 13, 12))
            # 照準
            pygame.draw.line(self.screen, (240, 240, 240), (W // 2 - 8, H // 2), (W // 2 + 8, H // 2), 2)
            pygame.draw.line(self.screen, (240, 240, 240), (W // 2, H // 2 - 8), (W // 2, H // 2 + 8), 2)
        if self.paused:
            shade = pygame.Surface((W, H), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 150))
            self.screen.blit(shade, (0, 0))
            self._btn_resume = pygame.Rect(W // 2 - 150, H // 2 - 50, 300, 40)
            self._btn_quit = pygame.Rect(W // 2 - 150, H // 2 + 10, 300, 40)
            for r, label in ((self._btn_resume, "Resume Game"), (self._btn_quit, "Save & Quit to Title")):
                pygame.draw.rect(self.screen, (120, 120, 120), r)
                pygame.draw.rect(self.screen, (230, 230, 230), r, 2)
                self.screen.blit(self.font.render(label, True, (255, 255, 255)), (r.x + 20, r.y + 12))
        else:
            self._btn_resume = self._btn_quit = pygame.Rect(-1, -1, 0, 0)
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        last_log = time.time()
        next_pause = self.now() + self.args.pause_every if self.args.pause_every else None
        running = True
        while running and self.now() < self.args.seconds:
            dt = clock.tick(self.args.fps) / 1000.0
            for ev in pygame.event.get():
                if not self.handle(ev):
                    running = False
            self.poll_mouse()
            # ユーザーが別ウィンドウへ切り替えた状況を模擬: ポーズメニューを開いてしばらくそのまま
            if next_pause and self.now() > next_pause:
                if not self.paused:
                    self.set_paused(True)
                    self._resume_at = self.now() + 3.0
                elif self.now() > getattr(self, "_resume_at", 0):
                    self.set_paused(False)
                    next_pause = self.now() + self.args.pause_every
            self.update(min(dt, 0.1))
            self.draw()
            self.frames += 1
            if time.time() - last_log >= 1.0:
                last_log = time.time()
                self.second_stats()
                self.dump()
        self.dump()
        pygame.quit()

    def dump(self):
        if not self.args.log:
            return
        out = dict(self.stats)
        out.update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.world.stats.items()})
        out["t"] = round(self.now(), 1)
        out["fps"] = round(self.frames / max(self.now(), 1e-3), 1)
        out["flying"] = self.flying
        with open(self.args.log, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--render-w", type=int, default=320)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slimes", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=1e9)
    ap.add_argument("--creative", action="store_true")
    ap.add_argument("--no-bobbing", dest="bobbing", action="store_false")
    ap.add_argument("--no-hud", dest="hud", action="store_false")
    ap.add_argument("--no-hand", dest="hand", action="store_false")
    ap.add_argument("--pause-every", type=float, default=0.0, help="この秒数ごとにポーズメニューを開く（フォーカス喪失の模擬）")
    ap.add_argument("--log")
    FakeCraft(ap.parse_args()).run()


if __name__ == "__main__":
    main()

"""コマンドライン: python -m flycraft <command>"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np

BANNER = r"""
   ___ _        ___           __ _
  | __| |_  _  / __|_ _ __ _ / _| |_    ハエの脳 × マインクラフト統合版
  | _|| | || || (__| '_/ _` |  _|  _|   FlyWire 全脳コネクトーム (139k neurons)
  |_| |_|\_, | \___|_| \__,_|_|  \__|
         |__/
"""


def _log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ helpers
def load_brain(args):
    """(connectome, retinotopy) を用意する。"""
    from . import data, retinotopy

    if getattr(args, "toy", False):
        from .toy import build_toy

        _log("⚠ トイ・コネクトーム（人工の小さな回路）で動かします。本物のハエの脳ではありません。")
        con = build_toy(seed=args.seed or 0)
        return con, retinotopy.infer(con)
    ddir = Path(args.data_dir) if args.data_dir else data.default_data_dir()
    if not data.has_flywire(ddir):
        _log(f"FlyWire のデータが見つかりません（{ddir}）。")
        if sys.stdin.isatty():
            ans = input("今ダウンロードしますか？ 約 135 MB [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes", "はい"):
                data.download(ddir, log=_log)
            else:
                _log("`--toy` を付けるとデータ無しで人工回路のデモを動かせます。")
                sys.exit(1)
        else:
            _log("`python -m flycraft download` を実行するか、`--toy` を付けてください。")
            sys.exit(1)
    t = time.time()
    con = data.load_flywire(ddir)
    _log(f"🧠 {con.name}: {con.n:,} ニューロン / {con.n_connections:,} 結合 / {con.n_synapses:,} シナプス"
         f"（{time.time() - t:.1f} 秒で読み込み）")
    retino = retinotopy.load_or_infer(con, ddir)
    _log(f"👁  網膜位相を推定: {int(retino.known.sum()):,} 個の視覚系ニューロンに視野上の位置を割り当て")
    return con, retino


def make_fly(args, con, retino):
    from .brain import LIFParams
    from .fly import Fly, FlyConfig

    lif = LIFParams(dt=args.dt)
    if args.no_std:
        lif.std_U = 0.0
    cfg = FlyConfig(hunger=args.hunger, acuity=args.acuity, retina=args.retina, seed=args.seed,
                    engine=args.engine, threads=args.threads, lif=lif)
    fly = Fly(con, retino, cfg)
    _log(f"⚙  シミュレーション: {fly.brain.engine} エンジン, dt = {lif.dt} ms"
         + ("" if fly.brain.engine == "numba" else "（pip install numba で数倍速くなります）"))
    return fly


def run_session(args, backend, fly) -> None:
    from .runner import Session

    session = Session(backend, fly, dt_ms=args.tick, realtime=not args.fast, log=_log)
    httpd = None
    if not args.no_dashboard:
        from .dashboard.server import serve

        try:
            httpd = serve(session, args.host, args.port)
            url = f"http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}/"
            _log(f"📊 ダッシュボード: {url}")
            if not args.no_browser:
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
        except OSError as e:
            _log(f"ダッシュボードを開始できません（ポート {args.port}）: {e}")
    session.start()
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    t0 = time.time()
    last = 0.0
    try:
        while not stop["flag"] and session.running:
            time.sleep(0.2)
            if args.seconds and fly.brain.t_ms / 1000.0 >= args.seconds:
                break
            if time.time() - last > 5:
                last = time.time()
                a = session.action
                _log(f"  t={fly.brain.t_ms / 1000:6.1f}s  速度 {fly.realtime_factor():.2f}×  "
                     f"前進 {a.forward:+.2f} 旋回 {a.turn:+.2f}  ジャンプ {fly.stats['jumps']}  "
                     f"噛む {fly.stats['bites']}  空腹 {fly.hunger:.2f}")
    finally:
        session.stop()
        if httpd:
            httpd.shutdown()
        extra = backend.extra_telemetry().get("game_stats")
        _log(f"終了: 脳の時間 {fly.brain.t_ms / 1000:.1f} 秒 / 実時間 {time.time() - t0:.1f} 秒"
             + (f" / {extra}" if extra else ""))


# ----------------------------------------------------------------- commands
def cmd_download(args) -> None:
    from . import data

    path = data.download(Path(args.data_dir) if args.data_dir else None, force=args.force, log=_log)
    _log(f"完了: {path}")


def cmd_sim(args) -> None:
    from .backends.sim import SimWorld

    con, retino = load_brain(args)
    fly = make_fly(args, con, retino)
    world = SimWorld(seed=args.seed or 0, fov_v=args.fov, width=args.width, height=args.height,
                     n_slimes=args.slimes)
    _log("🌍 内蔵ワールドで開始します（Ctrl+C で終了）")
    run_session(args, world, fly)


def cmd_bedrock(args) -> None:
    from .backends.bedrock_ws import BedrockWSBackend

    con, retino = load_brain(args)
    fly = make_fly(args, con, retino)
    backend = BedrockWSBackend(host=args.ws_host, port=args.ws_port, fov_v=args.fov,
                               screen_vision=args.screen_vision, window=args.window, log=_log)
    run_session(args, backend, fly)


def cmd_screen(args) -> None:
    from .backends.screen import ScreenBackend

    con, retino = load_brain(args)
    fly = make_fly(args, con, retino)
    backend = ScreenBackend(window=args.window, region=args.region, fov_v=args.fov,
                            mouse_speed=args.mouse_speed, log=_log)
    run_session(args, backend, fly)


def cmd_probe(args) -> None:
    """in silico 実験: ニューロン群を刺激して、下行性ニューロンの応答を表示する。"""
    from .brain import Brain, LIFParams
    from .fly import OPTO_GROUPS
    from .motor import MotorDecoder

    con, _ = load_brain(args)
    if args.stim in OPTO_GROUPS:
        g = OPTO_GROUPS[args.stim]
        sel, mode, value = g["select"], g["mode"], g["value"]
    else:
        sel, mode, value = {"cell_type": args.stim}, "rate", args.rate
        if args.side:
            sel["side"] = args.side
    idx = con.select(**sel)
    if not len(idx):
        _log(f"該当するニューロンがありません: {sel}")
        sys.exit(1)
    lif = LIFParams(dt=args.dt)
    if args.no_std:
        lif.std_U = 0.0
    brain = Brain(con, lif, engine=args.engine, seed=args.seed)
    if mode == "rate":
        brain.set_drive(idx, np.full(len(idx), value))
    else:
        brain.set_bias(idx, value)
    t = time.time()
    counts = brain.run(args.ms)
    el = time.time() - t
    hz = counts / (args.ms / 1000.0)
    _log(f"刺激: {sel} ({len(idx)} 個, {mode}={value}) を {args.ms:.0f} ms（計算 {el:.1f} 秒）")
    _log(f"発火したニューロン: {int((counts > 0).sum()):,} 個")
    dec = MotorDecoder(con)
    _log("\n運動出力ニューロン [Hz]:")
    for key, ids in dec.neurons().items():
        if len(ids):
            _log(f"  {key:16s} " + " ".join(f"{hz[i]:6.1f}" for i in ids))
    dn = con.select(super_class="descending")
    top = dn[np.argsort(-hz[dn])][: args.top]
    _log(f"\n最も活動した下行性ニューロン（上位 {args.top}）:")
    for i in top:
        if hz[i] <= 0:
            break
        _log(f"  {con.ann['cell_type'][i] or '?':12s} {con.ann['side'][i]:6s} {hz[i]:7.1f} Hz")


def cmd_bench(args) -> None:
    from .brain import Brain, LIFParams

    con, _ = load_brain(args)
    brain = Brain(con, LIFParams(dt=args.dt), engine=args.engine, seed=0, threads=args.threads)
    idx = con.select(super_class="visual_projection")[::4]
    brain.set_drive(idx, np.full(len(idx), 30.0))
    brain.run(20)
    t = time.time()
    brain.run(2000)
    el = time.time() - t
    _log(f"{brain.engine}: 脳の 2.0 秒を {el:.2f} 秒で計算 → 実時間の {2.0 / el:.2f} 倍")


# --------------------------------------------------------------------- main
def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("脳")
    g.add_argument("--toy", action="store_true", help="FlyWire の代わりにトイ・コネクトーム（人工回路）を使う")
    g.add_argument("--data-dir", help="データの置き場所（既定: ~/.cache/flycraft、環境変数 FLYCRAFT_DATA）")
    g.add_argument("--engine", default="auto", choices=["auto", "numba", "numpy"])
    g.add_argument("--threads", type=int, help="numba のスレッド数")
    g.add_argument("--dt", type=float, default=0.5, help="積分ステップ [ms]（1.0 にすると約 2 倍速い）")
    g.add_argument("--no-std", action="store_true", help="短期シナプス抑圧を無効化（元の Shiu et al. モデル）")
    g.add_argument("--seed", type=int, default=None)


def _run_opts(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("実行")
    g.add_argument("--hunger", type=float, default=0.6, help="初期の空腹度 0〜1（高いほどよく歩く）")
    g.add_argument("--acuity", type=float, default=3.0, help="複眼の解像度 [度]（実際のハエは約 5°）")
    g.add_argument("--retina", action="store_true", help="視細胞も直接駆動する")
    g.add_argument("--fov", type=float, default=70.0, help="ゲームの視野角（垂直, 度）")
    g.add_argument("--tick", type=float, default=50.0, help="1 ループで進める時間 [ms]")
    g.add_argument("--fast", action="store_true", help="（内蔵ワールド）実時間に合わせず最速で回す")
    g.add_argument("--seconds", type=float, default=0, help="脳の時間でこの秒数だけ動かして終了")
    g.add_argument("--port", type=int, default=8765, help="ダッシュボードのポート")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--no-dashboard", action="store_true")
    g.add_argument("--no-browser", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="flycraft", description="ハエの全脳コネクトームにマインクラフト統合版をプレイさせる")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="FlyWire 全脳コネクトームを取得する（約 135 MB）")
    p.add_argument("--data-dir")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("sim", help="内蔵のボクセルワールドで遊ばせる（Minecraft 不要）")
    _common(p)
    _run_opts(p)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--height", type=int, default=108)
    p.add_argument("--slimes", type=int, default=4)
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser("bedrock", help="統合版に /connect で接続して遊ばせる（WebSocket）")
    _common(p)
    _run_opts(p)
    p.add_argument("--ws-host", default="0.0.0.0")
    p.add_argument("--ws-port", type=int, default=19131)
    p.add_argument("--screen-vision", action="store_true",
                   help="視覚を画面キャプチャから得る（同じ PC で統合版を動かしている場合）")
    p.add_argument("--window", default="Minecraft", help="画面キャプチャするウィンドウ名")
    p.set_defaults(func=cmd_bedrock)

    p = sub.add_parser("screen", help="画面を見てキーボード・マウスで操作する（Windows 推奨）")
    _common(p)
    _run_opts(p)
    p.add_argument("--window", default="Minecraft", help="対象ウィンドウ名（部分一致）")
    p.add_argument("--region", help="キャプチャ範囲 x,y,w,h（ウィンドウが見つからない場合）")
    p.add_argument("--mouse-speed", type=float, default=6.0, help="旋回 1.0 あたりのマウス移動量 [px/tick]")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("probe", help="in silico 実験: ニューロン群を刺激して応答を見る")
    _common(p)
    p.add_argument("stim", help="刺激する群（sugar, bitter, lplc2, gf, p9, mdn ...）または細胞タイプ名（例: LC10a）")
    p.add_argument("--side", choices=["left", "right"])
    p.add_argument("--rate", type=float, default=100.0, help="細胞タイプ指定時のポアソン発火率 [Hz]")
    p.add_argument("--ms", type=float, default=1000.0)
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("bench", help="脳シミュレーションの速度を測る")
    _common(p)
    p.set_defaults(func=cmd_bench)
    return ap


def main(argv: Optional[list] = None) -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd in ("sim", "bedrock", "screen"):
        _log(BANNER)
    args.func(args)


if __name__ == "__main__":
    main()

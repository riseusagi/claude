"""ダッシュボード用の小さな HTTP サーバー（標準ライブラリのみ）。"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np

from ..fly import OPTO_GROUPS
from ..runner import Session

HERE = Path(__file__).resolve().parent

CLASS_COLORS = {
    "optic": "#4c8dff",
    "visual_projection": "#35d0e0",
    "visual_centrifugal": "#6fb6ff",
    "central": "#b07cff",
    "sensory": "#5fe07a",
    "sensory_ascending": "#9be07a",
    "ascending": "#f2d45c",
    "descending": "#ff9a3d",
    "motor": "#ff5b5b",
    "endocrine": "#ff7fd1",
    "": "#aaaaaa",
}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def brain_layout(session: Session, width: int = 900) -> dict:
    """ニューロンの正面図座標（x: 左右, y: 背腹）と分類。"""
    con = session.fly.con
    pos = con.pos
    ok = np.any(pos != 0, axis=1)
    x, y = pos[:, 0], pos[:, 1]
    if ok.any():
        x0, x1 = np.percentile(x[ok], [0.2, 99.8])
        y0, y1 = np.percentile(y[ok], [0.2, 99.8])
    else:
        x0, x1, y0, y1 = 0, 1, 0, 1
    height = int(width * (y1 - y0) / max(1e-6, (x1 - x0))) or width // 2
    px = np.clip((x - x0) / max(1e-6, x1 - x0) * (width - 1), 0, width - 1).astype(np.int16)
    py = np.clip((y - y0) / max(1e-6, y1 - y0) * (height - 1), 0, height - 1).astype(np.int16)
    px[~ok] = -1
    names, codes = np.unique(con.ann["super_class"], return_inverse=True)
    motor = {k: v.tolist() for k, v in session.fly.motor.neurons().items()}
    return {
        "n": int(con.n),
        "name": con.name,
        "w": width,
        "h": height,
        "x": _b64(px),
        "y": _b64(py),
        "cls": _b64(codes.astype(np.uint8)),
        "classes": [str(s) for s in names],
        "colors": [CLASS_COLORS.get(str(s), "#aaaaaa") for s in names],
        "motor": motor,
        "opto": {k: g["label"] for k, g in OPTO_GROUPS.items()},
    }


class _Handler(BaseHTTPRequestHandler):
    session: Session = None  # type: ignore
    layout_json: Optional[bytes] = None

    def log_message(self, fmt, *args):  # 静かに
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            snap = self.session.snapshot()
            self._send(200, json.dumps(snap, ensure_ascii=False).encode("utf-8"), "application/json")
        elif path == "/api/brain":
            cls = type(self)
            if cls.layout_json is None:
                cls.layout_json = json.dumps(brain_layout(self.session), ensure_ascii=False).encode()
            self._send(200, cls.layout_json, "application/json")
        elif path == "/api/minimap":
            be = self.session.backend
            if hasattr(be, "minimap"):
                img = be.minimap()
                body = json.dumps({"w": img.shape[1], "h": img.shape[0], "rgb": _b64(img),
                                   "size": getattr(be, "size", img.shape[1])})
                self._send(200, body.encode(), "application/json")
            else:
                self._send(404, b"{}", "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/api/control":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            msg = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b"bad json", "text/plain")
            return
        if isinstance(msg, dict):
            self.session.control(msg)
        self._send(200, b"{}", "application/json")


def serve(session: Session, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"session": session, "layout_json": None})
    httpd = ThreadingHTTPServer((host, port), handler)
    th = threading.Thread(target=httpd.serve_forever, name="flycraft-dashboard", daemon=True)
    th.start()
    return httpd

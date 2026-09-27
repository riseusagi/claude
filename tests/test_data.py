import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from flycraft import data

PAYLOAD = bytes(range(256)) * 4000  # 1 MB


class _Flaky(BaseHTTPRequestHandler):
    """最初の全体取得だけ途中で接続を切り、Range 要求には正しく応える。"""

    calls = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).calls += 1
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            body = PAYLOAD[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not rng and type(self).calls == 1:
            self.wfile.write(body[: len(body) // 3])  # 途中で切る
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


def test_fetch_resumes_after_truncation(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Flaky)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        dest = tmp_path / "file.bin"
        data._fetch(f"http://127.0.0.1:{httpd.server_address[1]}/f", dest, log=lambda *_: None)
        assert dest.read_bytes() == PAYLOAD
        assert _Flaky.calls >= 2
    finally:
        httpd.shutdown()

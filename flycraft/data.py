"""FlyWire 全脳コネクトーム（v783）のダウンロードとキャッシュ。

データの出典（ライセンス・引用は README を参照）:

* 結合行列: Shiu et al. 2024 "A Drosophila computational brain model reveals
  sensorimotor processing" (Nature) の公開リポジトリ
  (github.com/philshiu/Drosophila_brain_model) にある FlyWire v783 の派生ファイル。
* 細胞注釈: Schlegel et al. 2024 (Nature) の flywire_annotations。
* 元データ: Dorkenwald et al. 2024 (Nature), FlyWire Consortium.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .connectome import Connectome

CACHE_NAME = "flywire_783.npz"


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    filename: str
    sha256: str  # 開発時点のハッシュ。上流が更新されていたら警告のみ。
    size_hint: str


SOURCES = (
    Source(
        "completeness",
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv",
        "Completeness_783.csv",
        "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
        "3 MB",
    ),
    Source(
        "connectivity",
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
        "Connectivity_783.parquet",
        "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
        "100 MB",
    ),
    Source(
        "annotations",
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
        "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        "Supplemental_file1_neuron_annotations.tsv",
        "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be",
        "32 MB",
    ),
)

# FlyWire (FAFB) のボクセルサイズ [nm]。注釈の pos_x/y/z はこの単位。
VOXEL_NM = np.array([4.0, 4.0, 40.0], dtype=np.float64)


def default_data_dir() -> Path:
    env = os.environ.get("FLYCRAFT_DATA")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "flycraft"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "flycraft"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_once(url: str, part: Path, log=print) -> bool:
    """1 回分の取得。途中まであれば Range で続きから。完了したら True。"""
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "flycraft"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # 続きからに対応していない → 最初から
            have = 0
        length = int(r.headers.get("Content-Length") or 0)
        total = have + length if length else 0
        done = have
        last = -1
        with open(part, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct // 10 != last // 10:
                        log(f"    {part.name[:-5]}: {pct}% ({done / 1e6:.0f}/{total / 1e6:.0f} MB)")
                        last = pct
    return not total or done == total


def _fetch(url: str, dest: Path, log=print, attempts: int = 5) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    for k in range(attempts):
        try:
            if _fetch_once(url, part, log=log):
                shutil.move(str(part), str(dest))
                return
            log("    …通信が途中で切れました。続きから再開します")
        except (OSError, http.client.HTTPException) as e:
            log(f"    …取得に失敗しました（{e}）。再試行します")
    raise IOError(f"{url} を取得できませんでした（{attempts} 回失敗）")


def _is_complete(dest: Path, src: "Source") -> bool:
    marker = dest.with_suffix(dest.suffix + ".ok")
    if not dest.exists():
        return False
    if marker.exists():
        return True
    if _sha256(dest) == src.sha256:
        marker.touch()
        return True
    return False


def download(data_dir: Optional[Path] = None, force: bool = False, log=print) -> Path:
    """必要なファイルを取得して、シミュレーション用キャッシュ (npz) を作る。"""
    data_dir = Path(data_dir or default_data_dir())
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        dest = raw / src.filename
        marker = dest.with_suffix(dest.suffix + ".ok")
        if not force and _is_complete(dest, src):
            log(f"  ✓ {src.filename}（取得済み）")
            continue
        log(f"  ↓ {src.filename} ({src.size_hint}) を取得中…")
        _fetch(src.url, dest, log=log)
        if _sha256(dest) != src.sha256:
            log(f"  ! {src.filename} のハッシュが開発時点と異なります（上流が更新された可能性）。続行します。")
        marker.touch()
    cache = data_dir / CACHE_NAME
    if force or not cache.exists():
        log("  ⚙ 結合行列を変換中（初回のみ・数十秒）…")
        try:
            con = build_from_raw(raw)
        except Exception as e:
            raise RuntimeError(
                f"データの変換に失敗しました（{e}）。ファイルが壊れている可能性があります。"
                "`python -m flycraft download --force` で取り直してください。"
            ) from e
        con.save(cache)
        log(f"  ✓ {cache} を作成: {con.n:,} ニューロン, {con.n_connections:,} 結合, {con.n_synapses:,} シナプス")
    return cache


def build_from_raw(raw_dir: Path) -> Connectome:
    """生データ (csv/parquet/tsv) から Connectome を構築する。"""
    import pandas as pd

    raw_dir = Path(raw_dir)
    comp = pd.read_csv(raw_dir / "Completeness_783.csv", index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)

    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
    edges = pd.read_parquet(raw_dir / "Connectivity_783.parquet", columns=cols)
    pre = edges[cols[0]].to_numpy(np.int64)
    post = edges[cols[1]].to_numpy(np.int64)
    w = edges[cols[2]].to_numpy(np.float32)
    del edges

    ann_df = pd.read_csv(
        raw_dir / "Supplemental_file1_neuron_annotations.tsv",
        sep="\t",
        low_memory=False,
        usecols=["root_id", "pos_x", "pos_y", "pos_z", "super_class", "cell_class",
                 "cell_sub_class", "cell_type", "side"],
    )
    ann_df = ann_df.drop_duplicates("root_id").set_index("root_id").reindex(root_ids)
    ann = {}
    for col in ("super_class", "cell_class", "cell_sub_class", "cell_type", "side"):
        ann[col] = ann_df[col].fillna("").astype(str).to_numpy(dtype=str)
    pos = ann_df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64) * VOXEL_NM / 1000.0
    pos = np.nan_to_num(pos, nan=0.0).astype(np.float32)

    return Connectome.from_edges(pre, post, w, root_ids, ann=ann, pos=pos, name="FlyWire v783")


def load_flywire(data_dir: Optional[Path] = None) -> Connectome:
    data_dir = Path(data_dir or default_data_dir())
    cache = data_dir / CACHE_NAME
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} がありません。先に `python -m flycraft download` を実行してください"
            "（約 135 MB のダウンロード）。"
        )
    return Connectome.load(cache)


def has_flywire(data_dir: Optional[Path] = None) -> bool:
    return (Path(data_dir or default_data_dir()) / CACHE_NAME).exists()

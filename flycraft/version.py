"""どのコードで動いているかを調べる（古いインストール版で動いていないかの確認用）。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import __version__

PKG_DIR = Path(__file__).resolve().parent


def git_commit() -> str:
    """このパッケージが git の作業ツリー内にあれば、そのコミット（短い形）。無ければ空文字。"""
    repo = PKG_DIR.parent
    if not (repo / ".git").exists():
        return ""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip()
    except Exception:
        return ""


def describe() -> str:
    c = git_commit()
    return f"{__version__}" + (f" ({c})" if c else " (インストール版)")


def is_installed_copy() -> bool:
    """git の作業ツリーではなく site-packages などにコピーされたコードで動いているか。"""
    return not (PKG_DIR.parent / ".git").exists()

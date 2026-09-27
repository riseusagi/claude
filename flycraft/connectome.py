"""コネクトーム（神経回路の配線図）のデータ構造。

``Connectome`` はニューロンの注釈（細胞タイプ・左右・位置）と、
シナプス前ニューロンごとの CSR 形式の結合行列を持つ。
重みは「符号付きシナプス数」（興奮性 +、抑制性 −）で、
Shiu et al. 2024 (Nature) の全脳 LIF モデルと同じ表現。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

import numpy as np

StrOrList = Union[str, Sequence[str], None]

ANNOTATION_COLUMNS = ("super_class", "cell_class", "cell_sub_class", "cell_type", "side")


@dataclass
class Connectome:
    """全ニューロンの配線と注釈。

    Attributes
    ----------
    root_ids : (N,) int64      FlyWire の root ID
    indptr   : (N+1,) int64    CSR 行ポインタ（行 = シナプス前ニューロン）
    indices  : (nnz,) int32    シナプス後ニューロンのインデックス
    weights  : (nnz,) float32  符号付きシナプス数
    ann      : 注釈列名 -> (N,) str 配列
    pos      : (N,3) float32   ニューロンの代表点 [µm]（x: 左→右, y: 背→腹, z: 前→後）
    """

    root_ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    ann: dict = field(default_factory=dict)
    pos: Optional[np.ndarray] = None
    name: str = "connectome"

    def __post_init__(self) -> None:
        n = len(self.root_ids)
        if len(self.indptr) != n + 1:
            raise ValueError("indptr の長さが N+1 ではありません")
        for col in ANNOTATION_COLUMNS:
            if col not in self.ann:
                self.ann[col] = np.full(n, "", dtype="<U1")
        if self.pos is None:
            self.pos = np.zeros((n, 3), dtype=np.float32)
        self._rid_index = None

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return len(self.root_ids)

    @property
    def n_synapses(self) -> int:
        return int(np.abs(self.weights).sum())

    @property
    def n_connections(self) -> int:
        return len(self.indices)

    def index_of_root(self, root_ids: Iterable[int]) -> np.ndarray:
        """root ID → ニューロンのインデックス（見つからないものは除外）。"""
        if self._rid_index is None:
            self._rid_index = {int(r): i for i, r in enumerate(self.root_ids)}
        out = [self._rid_index[int(r)] for r in root_ids if int(r) in self._rid_index]
        return np.asarray(out, dtype=np.int64)

    def select(
        self,
        cell_type: StrOrList = None,
        side: StrOrList = None,
        super_class: StrOrList = None,
        cell_class: StrOrList = None,
        cell_sub_class: StrOrList = None,
        root_ids: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        """注釈の完全一致でニューロンを選ぶ。複数条件は AND、リストは OR。"""
        mask = np.ones(self.n, dtype=bool)
        for col, want in (
            ("cell_type", cell_type),
            ("side", side),
            ("super_class", super_class),
            ("cell_class", cell_class),
            ("cell_sub_class", cell_sub_class),
        ):
            if want is None:
                continue
            if isinstance(want, str):
                want = [want]
            mask &= np.isin(self.ann[col], list(want))
        idx = np.flatnonzero(mask)
        if root_ids is not None:
            idx = np.intersect1d(idx, self.index_of_root(root_ids))
        return idx.astype(np.int64)

    def describe(self, i: int) -> str:
        a = self.ann
        parts = [a["cell_type"][i] or "?", a["side"][i] or "", a["super_class"][i] or ""]
        return " / ".join(p for p in parts if p)

    def out_degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def to_dense(self) -> np.ndarray:
        """小さなコネクトーム用（テスト向け）。W[pre, post]"""
        w = np.zeros((self.n, self.n), dtype=np.float32)
        for pre in range(self.n):
            s, e = self.indptr[pre], self.indptr[pre + 1]
            w[pre, self.indices[s:e]] = self.weights[s:e]
        return w

    # --------------------------------------------------------------- building
    @classmethod
    def from_edges(
        cls,
        pre: np.ndarray,
        post: np.ndarray,
        weight: np.ndarray,
        root_ids: np.ndarray,
        ann: Optional[dict] = None,
        pos: Optional[np.ndarray] = None,
        name: str = "connectome",
    ) -> "Connectome":
        """エッジリストから CSR を組み立てる（同じ pre→post は合算）。"""
        n = len(root_ids)
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        weight = np.asarray(weight, dtype=np.float32)
        if len(pre):
            key = pre * n + post
            order = np.argsort(key, kind="stable")
            key, weight = key[order], weight[order]
            uniq, start = np.unique(key, return_index=True)
            weight = np.add.reduceat(weight, start) if len(start) else weight
            pre, post = uniq // n, uniq % n
            keep = weight != 0
            pre, post, weight = pre[keep], post[keep], weight[keep]
        counts = np.bincount(pre, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return cls(
            root_ids=np.asarray(root_ids, dtype=np.int64),
            indptr=indptr,
            indices=post.astype(np.int32),
            weights=weight.astype(np.float32),
            ann=dict(ann or {}),
            pos=None if pos is None else np.asarray(pos, dtype=np.float32),
            name=name,
        )

    # -------------------------------------------------------------- save/load
    def save(self, path) -> None:
        arrays = {
            "root_ids": self.root_ids,
            "indptr": self.indptr,
            "indices": self.indices,
            "weights": self.weights,
            "pos": self.pos,
            "name": np.array(self.name),
        }
        for col, values in self.ann.items():
            arrays["ann_" + col] = np.asarray(values, dtype=str)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "Connectome":
        with np.load(path, allow_pickle=False) as z:
            ann = {k[4:]: z[k] for k in z.files if k.startswith("ann_")}
            return cls(
                root_ids=z["root_ids"],
                indptr=z["indptr"],
                indices=z["indices"],
                weights=z["weights"],
                ann=ann,
                pos=z["pos"],
                name=str(z["name"]),
            )

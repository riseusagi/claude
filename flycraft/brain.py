"""全脳スパイキングニューラルネットワーク（Leaky Integrate-and-Fire）。

基本モデルとパラメータは Shiu et al. 2024 (Nature) の全脳モデルに準拠:

    dv/dt = (v0 - v + g - a) / tau_m     （不応期中は停止）
    dg/dt = -g / tau_syn
    スパイク: v > v_th → v = v_reset, g = 0, 不応期 t_ref
    シナプス: シナプス前スパイクの t_delay 後に g += w_syn × 符号付きシナプス数 × x

感覚入力は Shiu et al. と同様に「ポアソン発火の強制」として与える
（元モデルでは 1 イベントで閾値を大きく超える入力を注入している）。

元モデルからの拡張（どちらも 0 にすれば元モデルと同一）:

* 短期シナプス抑圧 (STD, x): 元モデルは持続的な感覚入力を与え続けると、
  触角葉の興奮性局所ニューロンのループから全脳が 400 Hz で発火し続ける
  「てんかん様の暴走」に落ちる。実際のシナプスが持つ抑圧を入れると、
  短い刺激への応答（例: 糖受容体→口吻伸展運動ニューロン MN9）を保ったまま
  長時間安定して動かせる。感覚ニューロン自身には適用しない。
* スパイク頻度適応 (a): 既定では無効。

線形 ODE を 1 ステップごとに厳密解で積分し、シナプス伝達はスパイクした
ニューロンの出力だけを加算するイベント駆動方式。numba があれば JIT 版、
なければ numpy 版で動く（同じ更新規則）。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .connectome import Connectome

try:  # numba は任意依存（あれば数倍〜10倍速い）
    import numba as _nb

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - 環境依存
    _nb = None
    HAVE_NUMBA = False


# これより静止電位に近い膜電位・小さいシナプス電流は 0 とみなす（閾値まで 7 mV あるので
# 動態への影響は無視できる）。静止したニューロンを計算から外して高速化するため。
REST_V = 5e-3  # [mV]
REST_G = 1e-3  # [mV]
# 入力が来なければ、膜電位は今の値から最大でも「g × PSP_PEAK」しか上がらない
# （tau_m=20, tau_syn=5 のときの PSP の最大値の係数。g=1 のとき t≈9.2 ms で 0.1575）
PSP_PEAK = 0.1575


@dataclass
class LIFParams:
    dt: float = 0.5  # 積分ステップ [ms]
    v0: float = -52.0  # 静止電位 [mV]
    v_reset: float = -52.0  # リセット電位 [mV]
    v_th: float = -45.0  # 発火閾値 [mV]
    tau_m: float = 20.0  # 膜時定数 [ms]
    tau_syn: float = 5.0  # シナプス時定数 [ms]
    t_ref: float = 2.2  # 不応期 [ms]
    t_delay: float = 1.8  # シナプス遅延 [ms]
    w_syn: float = 0.275  # 1 シナプスあたりの重み [mV]
    # --- 拡張 ---
    std_U: float = 0.2  # 1 スパイクで消費される伝達資源の割合（0 で無効）
    tau_rec: float = 250.0  # 伝達資源の回復時定数 [ms]
    adapt_mV: float = 0.0  # 1 スパイクごとの適応電流の増分 [mV]（0 で無効）
    tau_adapt: float = 200.0  # 適応の時定数 [ms]

    def coefficients(self):
        a = math.exp(-self.dt / self.tau_m)
        c = math.exp(-self.dt / self.tau_syn)
        # u' = (-u + g)/tau_m, g = g0 e^{-t/tau_s} の厳密解における g の係数
        if abs(self.tau_syn - self.tau_m) < 1e-9:
            b = (self.dt / self.tau_m) * a
        else:
            b = self.tau_syn / (self.tau_syn - self.tau_m) * (c - a)
        return a, b, c

    @property
    def adapt_decay(self) -> float:
        return math.exp(-self.dt / self.tau_adapt) if self.tau_adapt > 0 else 0.0

    @property
    def delay_steps(self) -> int:
        return max(1, int(round(self.t_delay / self.dt)))

    @property
    def ref_steps(self) -> int:
        return max(0, int(round(self.t_ref / self.dt)))


class ThreadTuner:
    """計算にかかった時間を見ながら numba のスレッド数を選ぶ。

    ゲームと同じ PC で動かすと、空きコアより多いスレッドはかえって遅くなる。
    最初に候補を順に試して最速のものを選び、その後もときどき前後の数を試し直す。
    """

    def __init__(self, max_threads: int, trial_ms: float = 1500.0, revisit_ms: float = 30000.0) -> None:
        cands = [t for t in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if t <= max_threads]
        if max_threads not in cands:
            cands.append(max_threads)
        self.cands = cands
        self.trial_ms = trial_ms
        self.revisit_ms = revisit_ms
        self.score = {}  # スレッド数 → 実時間 / 脳の時間
        self.queue = list(reversed(cands))  # 多い順に試す
        self.current = self.queue.pop(0)
        self._acc_wall = 0.0
        self._acc_sim = 0.0
        self._since_revisit = 0.0

    def record(self, wall_s: float, sim_ms: float) -> int:
        self._acc_wall += wall_s
        self._acc_sim += sim_ms
        self._since_revisit += sim_ms
        if self._acc_sim < self.trial_ms:
            return self.current
        ratio = self._acc_wall / (self._acc_sim / 1000.0)
        old = self.score.get(self.current)
        self.score[self.current] = ratio if old is None else 0.5 * old + 0.5 * ratio
        self._acc_wall = self._acc_sim = 0.0
        if not self.queue and self._since_revisit > self.revisit_ms:
            best = self.best
            i = self.cands.index(best)
            self.queue = [self.cands[j] for j in (i - 1, i + 1) if 0 <= j < len(self.cands)]
            self._since_revisit = 0.0
        self.current = self.queue.pop(0) if self.queue else self.best
        return self.current

    @property
    def best(self) -> int:
        return min(self.score, key=self.score.get) if self.score else self.current


class Brain:
    """コネクトームから作るスパイキング全脳モデル。

    >>> brain = Brain(connectome)
    >>> brain.set_drive(indices, rates_hz)   # 感覚ニューロンのポアソン駆動
    >>> counts = brain.run(50.0)             # 50 ms 進めて各ニューロンのスパイク数を得る
    """

    def __init__(
        self,
        connectome: Connectome,
        params: Optional[LIFParams] = None,
        engine: str = "auto",
        seed: Optional[int] = None,
        threads: Optional[int] = None,
    ) -> None:
        self.con = connectome
        self.p = params or LIFParams()
        if engine == "auto":
            engine = "numba" if HAVE_NUMBA else "numpy"
        if engine in ("numba", "numba-lazy") and not HAVE_NUMBA:
            raise RuntimeError("numba がインストールされていません（pip install numba）")
        if engine not in ("numba", "numba-lazy", "numpy"):
            raise ValueError(f"unknown engine: {engine}")
        if engine == "numba-lazy" and self.p.adapt_mV != 0.0:
            engine = "numba"  # 遅延評価版はスパイク頻度適応に未対応
        self.engine = engine
        self.max_threads = _nb.config.NUMBA_NUM_THREADS if HAVE_NUMBA else 1
        self.threads = min(int(threads), self.max_threads) if threads else self.max_threads
        self._tuner: Optional[ThreadTuner] = None
        self.nchunks = 64 if connectome.n >= 4096 else 1
        self.n = connectome.n
        self.indptr = np.ascontiguousarray(connectome.indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(connectome.indices, dtype=np.int32)
        # 重みは mV 単位に換算しておく
        self.weights = np.ascontiguousarray(connectome.weights * self.p.w_syn, dtype=np.float32)
        # 短期抑圧の対象（感覚ニューロンは除外: 入力の強さをそのまま届ける）
        sc = connectome.ann["super_class"]
        self.std_U = np.where(
            np.isin(sc, ["sensory", "sensory_ascending"]), 0.0, self.p.std_U
        ).astype(np.float32)
        self.rng = np.random.default_rng(seed)
        self._drive_idx = np.zeros(0, dtype=np.int64)
        self._drive_rates = np.zeros(self.n, dtype=np.float32)
        self._silenced = np.zeros(self.n, dtype=np.uint8)
        self.bias = np.zeros(self.n, dtype=np.float32)  # 定常的な脱分極電流 [mV]
        self.reset()

    def autotune(self, on: bool = True) -> None:
        """スレッド数の自動調整（numba エンジンのみ）。"""
        self._tuner = ThreadTuner(self.max_threads) if (on and self.engine == "numba") else None
        if self._tuner:
            self.threads = self._tuner.current

    # ----------------------------------------------------------------- state
    def reset(self) -> None:
        n, p = self.n, self.p
        self.v = np.full(n, p.v0, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.rc = np.zeros(n, dtype=np.int32)  # 残り不応期ステップ
        self.wa = np.zeros(n, dtype=np.float32)  # 適応電流 [mV]
        self.xl = np.ones(n, dtype=np.float32)  # 直前のスパイク直後の伝達資源
        self.tl = np.full(n, -(10**9), dtype=np.int64)  # 直前のスパイクのステップ
        self.D = p.delay_steps + 1
        self.buf = np.zeros((self.D, n), dtype=np.float32)  # 遅延つきシナプス入力のリングバッファ
        # 静止状態でないニューロンの印（numba 版はこれが 0 のニューロンを読み飛ばす）
        self.act = np.zeros(n, dtype=np.uint8)
        # 遅延評価（engine="numba"）用: 最後に状態を更新したステップ、計算中の集合、到着予定の入力の宛先
        self.tupd = np.zeros(n, dtype=np.int64)
        self.hot = np.zeros(n, dtype=np.int32)
        self.is_hot = np.zeros(n, dtype=np.uint8)
        self.nhot = np.zeros(1, dtype=np.int64)
        self.touched = np.zeros((self.D, n), dtype=np.int32)
        self.tn = np.zeros(self.D, dtype=np.int64)
        self.stamp = np.full(n, -1, dtype=np.int64)
        self.t_ms = 0.0
        self._step = 0

    def resources(self) -> np.ndarray:
        """現在の伝達資源 x（1 = 抑圧なし）。"""
        if self.p.tau_rec <= 0:
            return np.ones(self.n, dtype=np.float32)
        el = (self._step - self.tl) * self.p.dt / self.p.tau_rec
        return (1.0 - (1.0 - self.xl) * np.exp(-el)).astype(np.float32)

    # ------------------------------------------------------------------ input
    def set_drive(self, idx: np.ndarray, rates_hz: np.ndarray) -> None:
        """ポアソン駆動のレート [Hz] を設定する（全体を置き換え）。"""
        rates = np.zeros(self.n, dtype=np.float32)
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx):
            np.maximum.at(rates, idx, np.asarray(rates_hz, dtype=np.float32))
        self.set_drive_dense(rates)

    def set_drive_dense(self, rates_hz: np.ndarray) -> None:
        rates = np.asarray(rates_hz, dtype=np.float32)
        self._drive_rates = rates
        self._drive_idx = np.flatnonzero(rates > 0).astype(np.int64)

    @property
    def drive_rates(self) -> np.ndarray:
        return self._drive_rates

    def set_bias(self, idx: np.ndarray, mV) -> None:
        """定常電流（静止電位からの定常偏位 [mV]）を与える。

        閾値との差 (v_th - v0 = 7 mV) を超えると周期的に発火し、抑制性入力で
        止めることもできる（強制スパイクと違い、脳内の抑制が効く）。
        負の値は過分極（抑制）。"""
        self.bias[np.asarray(idx, dtype=np.int64)] = mV

    def clear_bias(self) -> None:
        self.bias[:] = 0.0

    def silence(self, idx: np.ndarray, on: bool = True) -> None:
        """指定ニューロンを発火不能にする（in silico での抑制実験）。"""
        self._silenced[np.asarray(idx, dtype=np.int64)] = 1 if on else 0

    def _events(self, nsteps: int):
        """このチャンク分のポアソン入力イベント (ステップ, ニューロン) を生成する。"""
        idx = self._drive_idx
        if not len(idx):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        lam = self._drive_rates[idx].astype(np.float64) * (nsteps * self.p.dt / 1000.0)
        k = self.rng.poisson(lam)
        neurons = np.repeat(idx, k)
        steps = self.rng.integers(0, nsteps, len(neurons))
        order = np.argsort(steps, kind="stable")
        return steps[order].astype(np.int64), neurons[order].astype(np.int64)

    # -------------------------------------------------------------------- run
    def run(self, duration_ms: float) -> np.ndarray:
        """duration_ms だけ進め、期間中の各ニューロンのスパイク数 (int32) を返す。"""
        p = self.p
        nsteps = max(1, int(round(duration_ms / p.dt)))
        counts = np.zeros(self.n, dtype=np.int32)
        a, b, c = p.coefficients()
        ev_step, ev_neuron = self._events(nsteps)
        rec_steps = p.tau_rec / p.dt if p.tau_rec > 0 else 1e-9
        if self.engine == "numba-lazy":
            # 閾値に届きうるもの（と定常電流のあるもの）を最初の計算対象にする
            u = self.v - p.v0
            cand = np.flatnonzero((self.bias != 0.0) | (self.rc > 0)
                                  | (p.v0 + np.maximum(u, 0) + PSP_PEAK * np.maximum(self.g, 0) > p.v_th - 1e-4))
            coef = p.tau_syn / (p.tau_syn - p.tau_m) if abs(p.tau_syn - p.tau_m) > 1e-9 else 0.0
            _run_lazy(
                nsteps, self._step, self.v, self.g, self.rc, self.bias, self.xl, self.tl, self.tupd, self.buf,
                self.touched, self.tn, self.stamp, self.hot, self.is_hot, self.nhot, cand.astype(np.int64),
                p.delay_steps, self.indptr, self.indices, self.weights, ev_step, ev_neuron, self._silenced,
                self.std_U, counts, a, b, c, coef, p.v0, p.v_reset, p.v_th, p.ref_steps, rec_steps, PSP_PEAK,
            )
            self._sync(self._step + nsteps, a, c, coef)
        elif self.engine == "numba":
            _nb.set_num_threads(self.threads)
            t_start = time.perf_counter()
            # 外部から状態を書き換えた場合にも備えて、静止していないものを印付け
            self.act[(self.bias != 0.0) | (self.v != p.v0) | (self.g != 0.0)] = 1
            _run_numba(
                nsteps, self._step, self.v, self.g, self.rc, self.wa, self.bias, self.xl, self.tl, self.buf,
                self.act, p.delay_steps, self.indptr, self.indices, self.weights, ev_step, ev_neuron,
                self._silenced, self.std_U, counts, a, b, c, p.v0, p.v_reset, p.v_th,
                p.ref_steps, p.adapt_decay, p.adapt_mV, rec_steps, self.nchunks,
            )
            if self._tuner is not None:
                self.threads = self._tuner.record(time.perf_counter() - t_start, nsteps * p.dt)
        else:
            bounds = np.searchsorted(ev_step, np.arange(nsteps + 1))
            for k in range(nsteps):
                forced = ev_neuron[bounds[k]:bounds[k + 1]]
                self._step_numpy(self._step + k, forced, counts, a, b, c, rec_steps)
        self._step += nsteps
        self.t_ms += nsteps * p.dt
        return counts

    def _sync(self, step: int, a: float, c: float, coef: float) -> None:
        """計算対象外だったニューロンの状態を、厳密解で step 時点まで進める。"""
        n = step - self.tupd
        m = n > 0
        if m.any():
            k = n[m].astype(np.float64)
            A, C = a ** k, c ** k
            v, g = self.v[m].astype(np.float64), self.g[m].astype(np.float64)
            self.v[m] = (self.p.v0 + (v - self.p.v0) * A + g * coef * (C - A)).astype(np.float32)
            self.g[m] = (g * C).astype(np.float32)
        self.tupd[:] = step

    def _step_numpy(self, step, forced, counts, a, b, c, rec_steps) -> None:
        p = self.p
        slot = step % self.D
        g = self.g + self.buf[slot]
        self.buf[slot] = 0.0
        refr = self.rc > 0
        free = ~refr
        v, wa = self.v, self.wa
        vn = p.v0 + (v - p.v0) * a + g * b + (self.bias - wa) * (1.0 - a)
        gn = g * c
        v = np.where(free, vn, v).astype(np.float32)
        g = np.where(free, gn, g).astype(np.float32)
        spk = (v > p.v_th) & free
        if len(forced):
            spk[forced[free[forced]]] = True
        spk &= self._silenced == 0
        self.rc = np.where(refr, self.rc - 1, self.rc)
        if p.adapt_mV != 0.0:
            wa *= p.adapt_decay
            wa[wa <= 1e-6] = 0.0
        v[(np.abs(v - p.v0) < REST_V) & (self.bias == 0.0)] = p.v0
        g[np.abs(g) < REST_G] = 0.0
        sidx = np.flatnonzero(spk)
        if len(sidx):
            v[sidx] = p.v_reset
            g[sidx] = 0.0
            if p.adapt_mV != 0.0:
                wa[sidx] += p.adapt_mV
            self.rc[sidx] = p.ref_steps
            counts[sidx] += 1
            x = (1.0 - (1.0 - self.xl[sidx]) * np.exp(-(step - self.tl[sidx]) / rec_steps)).astype(
                np.float32
            )
            self.xl[sidx] = x - self.std_U[sidx] * x
            self.tl[sidx] = step
            starts = self.indptr[sidx]
            lens = self.indptr[sidx + 1] - starts
            total = int(lens.sum())
            if total:
                offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
                eidx = offs + np.arange(total)
                w = self.weights[eidx] * np.repeat(x, lens)
                tgt = (step + p.delay_steps) % self.D
                self.buf[tgt] += np.bincount(self.indices[eidx], weights=w, minlength=self.n).astype(
                    np.float32
                )
        self.v, self.g = v, g


# ---------------------------------------------------------------------- numba
if HAVE_NUMBA:

    @_nb.njit(cache=True, parallel=True, nogil=True)
    def _run_numba(nsteps, step0, v, g, rc, wa, bias, xl, tl, buf, act, dsteps, indptr, indices, weights,
                   ev_step, ev_neuron, silenced, std_u, counts, a, b, c, v0, vr, vth,
                   ref_steps, ad, adapt_inc, rec_steps, nchunks):  # pragma: no cover - JIT
        n = v.shape[0]
        D = buf.shape[0]
        L = (n + nchunks - 1) // nchunks
        forced = np.zeros(n, dtype=np.uint8)
        spk = np.empty(n, dtype=np.int64)
        spk_x = np.empty(n, dtype=np.float32)
        nspk = np.zeros(nchunks, dtype=np.int64)
        use_adapt = adapt_inc != 0.0
        ep = 0
        nev = ev_step.shape[0]
        for k in range(nsteps):
            step = step0 + k
            slot = step % D
            tgt = (step + dsteps) % D
            while ep < nev and ev_step[ep] == k:
                forced[ev_neuron[ep]] = 1
                act[ev_neuron[ep]] = 1
                ep += 1
            row = buf[slot]
            for ch in _nb.prange(nchunks):
                s0 = ch * L
                s1 = min(n, s0 + L)
                cnt = 0
                for i in range(s0, s1):
                    if act[i] == 0 and forced[i] == 0:
                        continue
                    gi = g[i] + row[i]
                    row[i] = 0.0
                    wi = 0.0
                    if use_adapt:
                        wi = wa[i]
                        if wi != 0.0:
                            wn = wi * ad
                            wa[i] = wn if wn > 1e-6 else 0.0
                    if rc[i] > 0:
                        rc[i] -= 1
                        g[i] = gi
                        forced[i] = 0
                        continue
                    vi = v[i]
                    fi = forced[i]
                    bi = bias[i]
                    if gi == 0.0 and vi == v0 and fi == 0 and wi == 0.0 and bi == 0.0:
                        # 静止状態: 先の遅延スロットに入力が無ければ非アクティブにする
                        pend = False
                        for d in range(D):
                            if buf[d, i] != 0.0:
                                pend = True
                        if not pend:
                            act[i] = 0
                        continue
                    vi = v0 + (vi - v0) * a + gi * b + (bi - wi) * (1.0 - a)
                    gi = gi * c
                    if (vi > vth or fi == 1) and silenced[i] == 0:
                        vi = vr
                        gi = 0.0
                        rc[i] = ref_steps
                        if use_adapt:
                            wa[i] += adapt_inc
                        counts[i] += 1
                        # 伝達資源は前回スパイクからの経過時間で解析的に回復させる
                        x = 1.0 - (1.0 - xl[i]) * np.exp(-(step - tl[i]) / rec_steps)
                        xl[i] = x - std_u[i] * x
                        tl[i] = step
                        spk[s0 + cnt] = i
                        spk_x[s0 + cnt] = x
                        cnt += 1
                    forced[i] = 0
                    if bi == 0.0 and abs(vi - v0) < REST_V:
                        vi = v0
                    if abs(gi) < REST_G:
                        gi = 0.0
                    v[i] = vi
                    g[i] = gi
                nspk[ch] = cnt
            out = buf[tgt]
            for ch in range(nchunks):
                s0 = ch * L
                for q in range(nspk[ch]):
                    i = spk[s0 + q]
                    sx = spk_x[s0 + q]
                    for e in range(indptr[i], indptr[i + 1]):
                        j = indices[e]
                        out[j] += weights[e] * sx
                        act[j] = 1

    @_nb.njit(cache=True, nogil=True)
    def _advance(i, k, v, g, tupd, apow, cpow, coef, v0):  # pragma: no cover - JIT
        n = k - tupd[i]
        if n > 0:
            if n < apow.shape[0]:
                A = apow[n]
                C = cpow[n]
            else:
                A = apow[1] ** n
                C = cpow[1] ** n
            vi = v0 + (v[i] - v0) * A + g[i] * coef * (C - A)
            v[i] = vi
            g[i] = g[i] * C
            tupd[i] = k

    @_nb.njit(cache=True, nogil=True)
    def _run_lazy(nsteps, step0, v, g, rc, bias, xl, tl, tupd, buf, touched, tn, stamp, hot, is_hot, nhot,
                  cand, dsteps, indptr, indices, weights, ev_step, ev_neuron, silenced, std_u, counts,
                  a, b, c, coef, v0, vr, vth, ref_steps, rec_steps, peak):  # pragma: no cover - JIT
        """遅延評価版（単一スレッド）。

        入力が届いたニューロンと、閾値に届きうるニューロン（hot）だけを毎ステップ積分し、
        それ以外は次に入力が届いたときに厳密解でまとめて進める。どのニューロンも
        入力が無い間に閾値を超えることはない（v + g × PSP_PEAK < v_th）ことを保証して
        計算対象から外すので、全ニューロンを毎ステップ積分するのと同じ結果になる。
        """
        n = v.shape[0]
        D = buf.shape[0]
        forced = np.zeros(n, dtype=np.uint8)
        spk = np.empty(n, dtype=np.int64)
        spk_x = np.empty(n, dtype=np.float32)
        # a^n, c^n の表（べき乗の計算を省く）
        apow = np.empty(4096)
        cpow = np.empty(4096)
        apow[0] = 1.0
        cpow[0] = 1.0
        for q in range(1, 4096):
            apow[q] = apow[q - 1] * a
            cpow[q] = cpow[q - 1] * c
        nh = nhot[0]
        for j in range(cand.shape[0]):
            i = cand[j]
            if is_hot[i] == 0:
                _advance(i, step0, v, g, tupd, apow, cpow, coef, v0)
                is_hot[i] = 1
                hot[nh] = i
                nh += 1
        ep = 0
        nev = ev_step.shape[0]
        for k in range(nsteps):
            step = step0 + k
            slot = step % D
            # 1) この時刻に届く入力
            for q in range(tn[slot]):
                i = touched[slot, q]
                if is_hot[i] == 0:
                    _advance(i, step, v, g, tupd, apow, cpow, coef, v0)
                    gi = g[i] + buf[slot, i]
                    g[i] = gi
                    buf[slot, i] = 0.0
                    # この入力を足しても閾値に届かないなら、計算対象に入れずに済ませる
                    u = v[i] - v0
                    if u < 0.0:
                        u = 0.0
                    gp = gi if gi > 0.0 else 0.0
                    if v0 + u + gp * peak > vth - 1e-4 or bias[i] != 0.0:
                        is_hot[i] = 1
                        hot[nh] = i
                        nh += 1
                else:
                    g[i] += buf[slot, i]
                    buf[slot, i] = 0.0
            tn[slot] = 0
            # 2) 感覚入力（強制スパイク）
            while ep < nev and ev_step[ep] == k:
                i = ev_neuron[ep]
                ep += 1
                if is_hot[i] == 0:
                    _advance(i, step, v, g, tupd, apow, cpow, coef, v0)
                    is_hot[i] = 1
                    hot[nh] = i
                    nh += 1
                forced[i] = 1
            # 3) 計算対象のニューロンを 1 ステップ積分
            nspk = 0
            idx = 0
            while idx < nh:
                i = hot[idx]
                gi = g[i]
                vi = v[i]
                bi = bias[i]
                if rc[i] > 0:
                    rc[i] -= 1
                    forced[i] = 0
                else:
                    vi = v0 + (vi - v0) * a + gi * b + bi * (1.0 - a)
                    gi = gi * c
                    if (vi > vth or forced[i] == 1) and silenced[i] == 0:
                        vi = vr
                        gi = 0.0
                        rc[i] = ref_steps
                        counts[i] += 1
                        x = 1.0 - (1.0 - xl[i]) * np.exp(-(step - tl[i]) / rec_steps)
                        xl[i] = x - std_u[i] * x
                        tl[i] = step
                        spk[nspk] = i
                        spk_x[nspk] = x
                        nspk += 1
                    forced[i] = 0
                    v[i] = vi
                    g[i] = gi
                tupd[i] = step + 1
                keep = rc[i] > 0 or bi != 0.0
                if not keep:
                    u = vi - v0
                    if u < 0.0:
                        u = 0.0
                    gp = gi if gi > 0.0 else 0.0
                    keep = v0 + u + gp * peak > vth - 1e-4
                if keep:
                    idx += 1
                else:
                    is_hot[i] = 0
                    nh -= 1
                    hot[idx] = hot[nh]
            # 4) スパイクを遅延つきで配る
            tstep = step + dsteps
            ts = tstep % D
            out = buf[ts]
            for s_ in range(nspk):
                i = spk[s_]
                sx = spk_x[s_]
                for e in range(indptr[i], indptr[i + 1]):
                    j = indices[e]
                    out[j] += weights[e] * sx
                    if stamp[j] != tstep:
                        stamp[j] = tstep
                        touched[ts, tn[ts]] = j
                        tn[ts] += 1
        nhot[0] = nh

else:  # pragma: no cover

    def _run_lazy(*args):
        raise RuntimeError("numba unavailable")

    def _run_numba(*args):
        raise RuntimeError("numba unavailable")

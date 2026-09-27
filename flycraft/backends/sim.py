"""内蔵のミニ・ボクセルワールド（Minecraft 風）。

Minecraft 本体が無くても、同じパイプライン（画面 → 複眼 → 脳 → 操作）を
試せるように、地形・木・花（甘い）・サボテン（苦い・痛い）・スライム（動く物体）
を持つ小さな世界と、一人称のレイキャスト描画・簡単な物理を実装する。
数値は Minecraft に合わせてある（歩行 4.3 ブロック/秒、ジャンプ 1.25 ブロックなど）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..interface import Action, Backend, Observation

try:
    import numba as _nb

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _nb = None
    HAVE_NUMBA = False

# ---------------------------------------------------------------- blocks
AIR, GRASS, DIRT, STONE, SAND, WATER, LOG, LEAVES, FLOWER, CACTUS, BEDROCK, PLANKS = range(12)
BLOCK_NAMES = ["air", "grass", "dirt", "stone", "sand", "water", "log", "leaves", "flower",
               "cactus", "bedrock", "planks"]
# 上面・側面の色 (RGB)
TOP = np.array([
    [0, 0, 0], [95, 159, 53], [134, 96, 67], [125, 125, 125], [219, 207, 163], [48, 88, 200],
    [160, 130, 80], [58, 122, 38], [235, 205, 40], [80, 130, 50], [60, 60, 60], [180, 140, 90],
], dtype=np.float32)
SIDE = np.array([
    [0, 0, 0], [115, 95, 60], [134, 96, 67], [125, 125, 125], [210, 198, 155], [48, 88, 200],
    [104, 83, 50], [52, 110, 34], [230, 60, 50], [70, 120, 45], [60, 60, 60], [170, 130, 82],
], dtype=np.float32)
SOLID = np.array([0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=np.uint8)
SKY_TOP = np.array([110, 160, 255], dtype=np.float32)
SKY_HORIZON = np.array([190, 215, 255], dtype=np.float32)

GRAVITY = 32.0  # ブロック/秒²
JUMP_V = 8.9  # 1.25 ブロック跳べる初速
WALK_SPEED = 4.3  # ブロック/秒
TURN_SPEED = 180.0  # 度/秒（旋回指令 1.0 のとき）
EYE = 1.62
HALF_W = 0.3
HEIGHT = 1.8


# ---------------------------------------------------------------- world gen
def _value_noise(shape, scale, rng):
    """なめらかな 2D ノイズ（格子点の乱数を双三次補間）。"""
    h, w = shape
    gh, gw = int(h / scale) + 3, int(w / scale) + 3
    grid = rng.random((gh, gw))
    y = np.arange(h) / scale
    x = np.arange(w) / scale
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    ty, tx = y - y0, x - x0
    sy, sx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
    g00 = grid[y0][:, x0]
    g01 = grid[y0][:, x0 + 1]
    g10 = grid[y0 + 1][:, x0]
    g11 = grid[y0 + 1][:, x0 + 1]
    top = g00 + (g01 - g00) * sx[None, :]
    bot = g10 + (g11 - g10) * sx[None, :]
    return top + (bot - top) * sy[:, None]


def generate_world(size: int = 96, height: int = 32, seed: int = 0) -> np.ndarray:
    """world[x, y, z] のブロック ID 配列を作る。y が上。"""
    rng = np.random.default_rng(seed)
    n = 0.6 * _value_noise((size, size), 24, rng) + 0.3 * _value_noise((size, size), 10, rng) \
        + 0.1 * _value_noise((size, size), 4, rng)
    w = np.zeros((size, height, size), dtype=np.uint8)
    w[:, 0, :] = BEDROCK
    hmap = (5 + n * 16).astype(int)
    for x in range(size):
        for z in range(size):
            h = hmap[x, z]
            w[x, 1:h - 3, z] = STONE
            w[x, h - 3:h, z] = DIRT
            w[x, h, z] = GRASS if h > 9 else SAND
    # 砂地の上には池
    water_level = 9
    for x in range(size):
        for z in range(size):
            if hmap[x, z] < water_level:
                w[x, hmap[x, z] + 1:water_level + 1, z] = WATER
    # 木
    for _ in range(size * size // 180):
        x, z = rng.integers(3, size - 3, 2)
        h = hmap[x, z]
        if w[x, h, z] != GRASS:
            continue
        th = rng.integers(4, 6)
        w[x, h + 1:h + 1 + th, z] = LOG
        top = h + th
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                for dy in range(-1, 2):
                    if abs(dx) + abs(dz) + max(0, dy) * 2 <= 3 and w[x + dx, top + dy, z + dz] == AIR:
                        w[x + dx, top + dy, z + dz] = LEAVES
    # 花（甘い）とサボテン（苦い）
    for _ in range(size * size // 80):
        x, z = rng.integers(1, size - 1, 2)
        h = hmap[x, z]
        if w[x, h, z] == GRASS and w[x, h + 1, z] == AIR:
            w[x, h + 1, z] = FLOWER
    for _ in range(size * size // 400):
        x, z = rng.integers(1, size - 1, 2)
        h = hmap[x, z]
        if w[x, h + 1, z] == AIR and w[x, h, z] in (GRASS, SAND):
            w[x, h + 1:h + 1 + rng.integers(1, 3), z] = CACTUS
    # 外周の壁（世界の端から落ちないように）
    w[0, :, :] = w[-1, :, :] = PLANKS
    w[:, :, 0] = w[:, :, -1] = PLANKS
    w[:, height - 1, :] = AIR
    return w


# ---------------------------------------------------------------- rendering
FLOWER_BOX = (0.3, 0.7, 0.0, 0.55)  # x/z の範囲と高さ


def _flower_hit_py(ox, oy, oz, dx, dy, dz, ix, iy, iz):
    """花（セル中央の小さな箱）とのレイ交差。戻り値 (t, 面)。"""
    lo = (ix + FLOWER_BOX[0], iy + FLOWER_BOX[2], iz + FLOWER_BOX[0])
    hi = (ix + FLOWER_BOX[1], iy + FLOWER_BOX[3], iz + FLOWER_BOX[1])
    o = (ox, oy, oz)
    d = (dx, dy, dz)
    tmin, tmax, face = -1e30, 1e30, 0
    for k in range(3):
        if d[k] == 0.0:
            if o[k] < lo[k] or o[k] > hi[k]:
                return np.inf, 0
            continue
        t1 = (lo[k] - o[k]) / d[k]
        t2 = (hi[k] - o[k]) / d[k]
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin, face = t1, k
        if t2 < tmax:
            tmax = t2
    if tmax >= tmin and tmin > 0:
        return tmin, face
    return np.inf, 0

def _render_py(world, ox, oy, oz, dirs, maxd, sky, top, side):
    """numba が無いとき用の numpy 版 DDA レイキャスト。"""
    sx, sy, sz = world.shape
    n = dirs.shape[0]
    dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    ix = np.full(n, int(math.floor(ox)))
    iy = np.full(n, int(math.floor(oy)))
    iz = np.full(n, int(math.floor(oz)))
    stepx, stepy, stepz = np.sign(dx).astype(int), np.sign(dy).astype(int), np.sign(dz).astype(int)
    with np.errstate(divide="ignore"):
        tdx = np.where(dx != 0, np.abs(1 / dx), 1e30)
        tdy = np.where(dy != 0, np.abs(1 / dy), 1e30)
        tdz = np.where(dz != 0, np.abs(1 / dz), 1e30)
    tmx = np.where(dx > 0, (ix + 1 - ox) * tdx, (ox - ix) * tdx)
    tmy = np.where(dy > 0, (iy + 1 - oy) * tdy, (oy - iy) * tdy)
    tmz = np.where(dz > 0, (iz + 1 - oz) * tdz, (oz - iz) * tdz)
    hit_t = np.full(n, np.inf)
    hit_b = np.zeros(n, dtype=np.int64)
    hit_f = np.zeros(n, dtype=np.int64)
    alive = np.ones(n, bool)
    t = np.zeros(n)
    for _ in range(int(maxd * 1.8) + 2):
        if not alive.any():
            break
        ax = np.argmin(np.stack([tmx, tmy, tmz]), axis=0)
        mx, my, mz = ax == 0, ax == 1, ax == 2
        t = np.where(mx, tmx, np.where(my, tmy, tmz))
        ix = ix + np.where(mx & alive, stepx, 0)
        iy = iy + np.where(my & alive, stepy, 0)
        iz = iz + np.where(mz & alive, stepz, 0)
        tmx = np.where(mx, tmx + tdx, tmx)
        tmy = np.where(my, tmy + tdy, tmy)
        tmz = np.where(mz, tmz + tdz, tmz)
        out = (ix < 0) | (iy < 0) | (iz < 0) | (ix >= sx) | (iy >= sy) | (iz >= sz) | (t > maxd)
        alive &= ~out
        b = np.zeros(n, dtype=np.int64)
        b[alive] = world[ix[alive], iy[alive], iz[alive]]
        fl = np.flatnonzero(alive & (b == FLOWER))
        for r in fl:
            tf, ff = _flower_hit_py(ox, oy, oz, dx[r], dy[r], dz[r], ix[r], iy[r], iz[r])
            if tf < np.inf:
                hit_t[r], hit_b[r], hit_f[r] = tf, FLOWER, ff
                alive[r] = False
        b[fl] = 0
        hit = alive & (b != 0)
        hit_t[hit] = t[hit]
        hit_b[hit] = b[hit]
        hit_f[hit] = np.where(mx, 0, np.where(my, 1, 2))[hit]
        alive &= ~hit
    return hit_t, hit_b, hit_f


if HAVE_NUMBA:

    _flower_hit = _nb.njit(cache=True)(_flower_hit_py)

    @_nb.njit(cache=True)
    def _cast_nb(world, ox, oy, oz, dx, dy, dz, maxd):  # pragma: no cover - JIT
        sx, sy, sz = world.shape
        ix, iy, iz = int(math.floor(ox)), int(math.floor(oy)), int(math.floor(oz))
        stx = 1 if dx > 0 else -1
        sty = 1 if dy > 0 else -1
        stz = 1 if dz > 0 else -1
        tdx = abs(1.0 / dx) if dx != 0 else 1e30
        tdy = abs(1.0 / dy) if dy != 0 else 1e30
        tdz = abs(1.0 / dz) if dz != 0 else 1e30
        tmx = ((ix + 1 - ox) if dx > 0 else (ox - ix)) * tdx
        tmy = ((iy + 1 - oy) if dy > 0 else (oy - iy)) * tdy
        tmz = ((iz + 1 - oz) if dz > 0 else (oz - iz)) * tdz
        for _ in range(4096):
            if tmx < tmy and tmx < tmz:
                t = tmx
                ix += stx
                tmx += tdx
                f = 0
            elif tmy < tmz:
                t = tmy
                iy += sty
                tmy += tdy
                f = 1
            else:
                t = tmz
                iz += stz
                tmz += tdz
                f = 2
            if t > maxd or ix < 0 or iy < 0 or iz < 0 or ix >= sx or iy >= sy or iz >= sz:
                return np.inf, 0, 0
            b = world[ix, iy, iz]
            if b == 8:  # 花は小さな箱として判定
                th, fh = _flower_hit(ox, oy, oz, dx, dy, dz, ix, iy, iz)
                if th < np.inf:
                    return th, 8, fh
            elif b != 0:
                return t, int(b), f
        return np.inf, 0, 0

    @_nb.njit(cache=True, parallel=True)
    def _render_nb(world, ox, oy, oz, dirs, maxd):  # pragma: no cover - JIT
        n = dirs.shape[0]
        hit_t = np.full(n, np.inf)
        hit_b = np.zeros(n, dtype=np.int64)
        hit_f = np.zeros(n, dtype=np.int64)
        for r in _nb.prange(n):
            t, b, f = _cast_nb(world, ox, oy, oz, dirs[r, 0], dirs[r, 1], dirs[r, 2], maxd)
            hit_t[r] = t
            hit_b[r] = b
            hit_f[r] = f
        return hit_t, hit_b, hit_f


# ---------------------------------------------------------------- entities
@dataclass
class Slime:
    x: float
    y: float
    z: float
    size: float = 0.9
    vx: float = 0.0
    vz: float = 0.0
    vy: float = 0.0
    hp: int = 4
    timer: float = 0.0
    color: tuple = (90, 200, 90)


@dataclass
class Player:
    x: float
    y: float
    z: float
    yaw: float = 0.0  # 度。0 = +z 方向, 正 = 左回り
    pitch: float = -8.0
    vy: float = 0.0
    on_ground: bool = False
    health: float = 20.0


# ---------------------------------------------------------------- backend
class SimWorld(Backend):
    """内蔵ワールド。step(action, dt) で 1 フレーム進めて観測を返す。"""

    name = "sim"

    def __init__(self, seed: int = 0, size: int = 96, width: int = 192, height: int = 108,
                 fov_v: float = 70.0, n_slimes: int = 5, max_dist: float = 48.0,
                 auto_step: bool = True) -> None:
        self.seed = seed
        self.size = size
        self.width, self.height = width, height
        self.fov_v = fov_v
        self.n_slimes = n_slimes
        self.max_dist = max_dist
        self.auto_step = auto_step
        self.rng = np.random.default_rng(seed + 1)
        self.world = generate_world(size, 32, seed)
        self._make_rays()
        self.t = 0.0
        self.events: List[str] = []
        self.stats = {"distance": 0.0, "flowers": 0, "damage": 0.0, "deaths": 0, "jumps": 0,
                      "blocks": 0, "bumps": 0, "slime_hits": 0}
        self._attack_time = 0.0
        self._sugar = 0.0
        self._bitter = 0.0
        self._touch = [0.0, 0.0]
        self.view_offset = [0.0, 0.0, 0.0, 0.0, 0.0]  # カメラのずれ (x, y, z, yaw, pitch)。視点の揺れの再現用
        self.reset()

    # ------------------------------------------------------------- helpers
    def _make_rays(self) -> None:
        fv = math.radians(self.fov_v)
        tv = math.tan(fv / 2)
        th = tv * self.width / self.height
        x = ((np.arange(self.width) + 0.5) / self.width * 2 - 1) * th
        y = (1 - (np.arange(self.height) + 0.5) / self.height * 2) * tv
        X, Y = np.meshgrid(x, y)
        d = np.stack([X, Y, np.ones_like(X)], -1)
        self._cam_rays = d / np.linalg.norm(d, axis=-1, keepdims=True)  # カメラ座標（x 右, y 上, z 前）

    @property
    def fov_h(self) -> float:
        return math.degrees(2 * math.atan(math.tan(math.radians(self.fov_v) / 2) * self.width / self.height))

    def _surface_y(self, x: int, z: int) -> int:
        col = self.world[x, :, z]
        solid = np.flatnonzero(SOLID[col] | (col == WATER))
        return int(solid.max()) + 1 if len(solid) else 1

    def _open_spot(self, x: int, y: int, z: int) -> bool:
        """地面があり、頭上が開けていて、周囲の半分以上へ歩いて（または 1 段登って）出られる場所。"""
        w = self.world
        if y + 4 >= w.shape[1] or w[x, y - 1, z] not in (GRASS, SAND, DIRT):
            return False
        if SOLID[w[x, y:y + 4, z]].any():
            return False
        free = 0
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            col = w[x + dx, :, z + dz]
            if not SOLID[col[y:y + 2]].any() or (not SOLID[col[y + 1:y + 4]].any()):
                free += 1
        return free >= 5

    def _spawn_point(self):
        c = self.size // 2
        for r in range(0, self.size // 2 - 2):
            for _ in range(20):
                x = int(np.clip(c + self.rng.integers(-r, r + 1), 2, self.size - 3))
                z = int(np.clip(c + self.rng.integers(-r, r + 1), 2, self.size - 3))
                y = self._surface_y(x, z)
                if self._open_spot(x, y, z):
                    return x + 0.5, float(y), z + 0.5
        return c + 0.5, 20.0, c + 0.5

    def reset(self) -> Observation:
        x, y, z = self._spawn_point()
        self.player = Player(x, y, z, yaw=float(self.rng.uniform(0, 360)))
        self.slimes = []
        for _ in range(self.n_slimes):
            sx, sy, sz = self._spawn_point()
            self.slimes.append(Slime(sx + self.rng.uniform(-6, 6), sy + 1, sz + self.rng.uniform(-6, 6)))
        return self._observe()

    # ------------------------------------------------------------- physics
    def _collides(self, x: float, y: float, z: float) -> bool:
        w = self.world
        x0, x1 = int(math.floor(x - HALF_W)), int(math.floor(x + HALF_W - 1e-6))
        y0, y1 = int(math.floor(y)), int(math.floor(y + HEIGHT - 1e-6))
        z0, z1 = int(math.floor(z - HALF_W)), int(math.floor(z + HALF_W - 1e-6))
        if x0 < 0 or z0 < 0 or x1 >= self.size or z1 >= self.size or y0 < 0:
            return True
        y1 = min(y1, w.shape[1] - 1)
        return bool(SOLID[w[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]].any())

    def _block_at(self, x: float, y: float, z: float) -> int:
        ix, iy, iz = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        if 0 <= ix < self.size and 0 <= iy < self.world.shape[1] and 0 <= iz < self.size:
            return int(self.world[ix, iy, iz])
        return AIR

    def _forward_vec(self, yaw: Optional[float] = None):
        a = math.radians(self.player.yaw if yaw is None else yaw)
        # yaw 0 で +z、正で左回り（上から見て反時計回り）→ 左 = +x
        return math.sin(a), math.cos(a)

    def _move(self, dx: float, dz: float) -> bool:
        """水平移動。ぶつかったら True。1 ブロックの段差は自動で登る（Bedrock の自動ジャンプ相当）。"""
        p = self.player
        bumped = False
        for axis, d in (("x", dx), ("z", dz)):
            if d == 0:
                continue
            nx, nz = (p.x + d, p.z) if axis == "x" else (p.x, p.z + d)
            if not self._collides(nx, p.y, nz):
                p.x, p.z = nx, nz
            elif self.auto_step and p.on_ground and not self._collides(nx, p.y + 1.0, nz) \
                    and not self._collides(p.x, p.y + 1.0, p.z):
                p.vy = JUMP_V
                p.on_ground = False
                bumped = True
            else:
                bumped = True
        return bumped

    def _damage(self, amount: float, why: str) -> None:
        p = self.player
        p.health -= amount
        self.stats["damage"] += amount
        self._bitter = max(self._bitter, min(1.0, 0.4 + amount / 4))
        self.events.append(f"痛い！({why})")
        if p.health <= 0:
            self.stats["deaths"] += 1
            self.events.append("力尽きた… リスポーン")
            x, y, z = self._spawn_point()
            p.x, p.y, p.z, p.vy, p.health = x, y, z, 0.0, 20.0

    def step(self, action: Action, dt: float) -> Observation:
        p = self.player
        self.t += dt
        self._sugar *= math.exp(-dt / 0.4)
        self._bitter *= math.exp(-dt / 0.4)
        self._touch = [v * math.exp(-dt / 0.3) for v in self._touch]
        # 旋回と歩行
        p.yaw = (p.yaw + TURN_SPEED * float(np.clip(action.turn, -1, 1)) * dt) % 360
        fx, fz = self._forward_vec()
        speed = WALK_SPEED * float(np.clip(action.forward, -1, 1))
        if speed < 0:
            speed *= 0.5
        x0, z0 = p.x, p.z
        in_water = self._block_at(p.x, p.y + 0.5, p.z) == WATER
        if in_water:
            speed *= 0.5
        bumped = self._move(fx * speed * dt, fz * speed * dt)
        self.stats["distance"] += math.hypot(p.x - x0, p.z - z0)
        if bumped and abs(speed) > 0.5:
            self.stats["bumps"] += 1
            # どちら側の剛毛が触れたか: 前方左右の壁を調べる
            lx, lz = self._forward_vec(p.yaw + 35)
            rx, rz = self._forward_vec(p.yaw - 35)
            sgn = 1 if speed > 0 else -1
            left = self._collides(p.x + sgn * lx * 0.4, p.y, p.z + sgn * lz * 0.4)
            right = self._collides(p.x + sgn * rx * 0.4, p.y, p.z + sgn * rz * 0.4)
            if not left and not right:
                left = right = True
            self._touch[0] = max(self._touch[0], 0.8 if left else 0.0)
            self._touch[1] = max(self._touch[1], 0.8 if right else 0.0)
        # ジャンプ
        if action.jump and (p.on_ground or in_water):
            p.vy = JUMP_V * (0.6 if in_water else 1.0)
            p.on_ground = False
            self.stats["jumps"] += 1
        # 重力
        g = GRAVITY * (0.3 if in_water else 1.0)
        # 放物運動を厳密に積分する（粗い刻みでもジャンプの高さが変わらないように）
        vmin = -40.0 if not in_water else -3.0
        v0 = p.vy
        p.vy = max(v0 - g * dt, vmin)
        ny = p.y + (v0 + p.vy) * 0.5 * dt
        if self._collides(p.x, ny, p.z):
            if p.vy < 0:
                fall = -p.vy
                p.y = math.floor(ny) + 1.0 if not self._collides(p.x, math.floor(ny) + 1.0, p.z) else p.y
                p.on_ground = True
                if fall > 20 and not in_water:
                    self._damage((fall - 20) / 3, "落下")
            p.vy = 0.0
        else:
            p.y = ny
            p.on_ground = self._collides(p.x, p.y - 0.05, p.z)
        # 足元・周囲の物体（花 = 甘い、サボテン = 苦い）
        for dx in (-HALF_W - 0.05, 0.0, HALF_W + 0.05):
            for dz in (-HALF_W - 0.05, 0.0, HALF_W + 0.05):
                for dy in (0.1, 1.0):
                    bx, by, bz = p.x + dx, p.y + dy, p.z + dz
                    b = self._block_at(bx, by, bz)
                    if b == FLOWER:
                        self.world[int(math.floor(bx)), int(math.floor(by)), int(math.floor(bz))] = AIR
                        self.stats["flowers"] += 1
                        self._sugar = 1.0
                        self.events.append("花の蜜を舐めた（甘い！）")
                    elif b == CACTUS and self.rng.random() < dt * 2:
                        self._damage(1.0, "サボテン")
        if p.y < -5:
            self._damage(100, "奈落")
        # 噛む（攻撃・採掘）
        if action.attack:
            self._attack_time += dt
            target = self._target()
            if target is not None:
                kind, obj = target
                if kind == "slime" and self._attack_time > 0.3:
                    obj.hp -= 1
                    obj.vy = 5.0
                    self._attack_time = 0.0
                    self.stats["slime_hits"] += 1
                    self.events.append("スライムを噛んだ")
                elif kind == "block" and self._attack_time > 0.6:
                    bx, by, bz, b = obj
                    if b not in (BEDROCK, PLANKS, WATER):
                        self.world[bx, by, bz] = AIR
                        self.stats["blocks"] += 1
                        self.events.append(f"{BLOCK_NAMES[b]} をかじり取った")
                        if b in (LEAVES, FLOWER):
                            self._sugar = max(self._sugar, 0.6)
                    self._attack_time = 0.0
        else:
            self._attack_time = 0.0
        self._update_slimes(dt)
        self.events = self.events[-20:]
        return self._observe()

    def _target(self):
        """視線の先 4 ブロック以内の対象（スライム優先）。"""
        p = self.player
        fx, fz = self._forward_vec()
        pr = math.radians(p.pitch)
        d = np.array([fx * math.cos(pr), math.sin(pr), fz * math.cos(pr)])
        eye = np.array([p.x, p.y + EYE, p.z])
        for s in self.slimes:
            c = np.array([s.x, s.y + s.size / 2, s.z])
            t = float(np.dot(c - eye, d))
            if 0 < t < 4.0 and np.linalg.norm(eye + d * t - c) < s.size * 0.8:
                return "slime", s
        for t in np.arange(0.2, 4.0, 0.1):
            q = eye + d * t
            b = self._block_at(*q)
            if b not in (AIR, WATER):
                return "block", (int(math.floor(q[0])), int(math.floor(q[1])), int(math.floor(q[2])), b)
        return None

    def _update_slimes(self, dt: float) -> None:
        p = self.player
        alive = []
        for s in self.slimes:
            if s.hp <= 0:
                self.events.append("スライムを倒した")
                continue
            s.timer -= dt
            onground = self._block_at(s.x, s.y - 0.05, s.z) not in (AIR, WATER, FLOWER)
            if onground and s.timer <= 0:
                dxp, dzp = p.x - s.x, p.z - s.z
                dist = math.hypot(dxp, dzp)
                if dist < 12 and self.rng.random() < 0.35:  # ときどきプレイヤーに向かって跳ねる
                    ang = math.atan2(dxp, dzp) + self.rng.normal(0, 0.3)
                else:
                    ang = self.rng.uniform(0, 2 * math.pi)
                sp = self.rng.uniform(2.0, 4.0)
                s.vx, s.vz = math.sin(ang) * sp, math.cos(ang) * sp
                s.vy = self.rng.uniform(5.0, 8.0)
                s.timer = self.rng.uniform(0.6, 2.0)
            s.vy -= GRAVITY * 0.8 * dt
            nx, ny, nz = s.x + s.vx * dt, s.y + s.vy * dt, s.z + s.vz * dt
            if self._block_at(nx, s.y + 0.1, s.z) in (AIR, WATER, FLOWER):
                s.x = nx
            if self._block_at(s.x, s.y + 0.1, nz) in (AIR, WATER, FLOWER):
                s.z = nz
            if self._block_at(s.x, ny, s.z) in (AIR, WATER, FLOWER):
                s.y = ny
            else:
                s.vy = 0.0
                s.vx *= 0.5
                s.vz *= 0.5
                s.y = math.floor(ny) + 1.0 if s.vy <= 0 else s.y
            s.x = float(np.clip(s.x, 1.5, self.size - 1.5))
            s.z = float(np.clip(s.z, 1.5, self.size - 1.5))
            if s.y < -5:
                continue
            if math.hypot(p.x - s.x, p.z - s.z) < 0.7 and abs((p.y + 0.9) - (s.y + s.size / 2)) < 1.2 \
                    and self.rng.random() < dt * 0.8:
                self._damage(1.0, "スライム")
                self._touch = [1.0, 1.0]
            alive.append(s)
        while len(alive) < self.n_slimes:
            sx, sy, sz = self._spawn_point()
            ang = self.rng.uniform(0, 2 * math.pi)
            alive.append(Slime(float(np.clip(sx + 14 * math.sin(ang), 2, self.size - 2)), sy + 2,
                               float(np.clip(sz + 14 * math.cos(ang), 2, self.size - 2))))
        self.slimes = alive

    # ------------------------------------------------------------ observation
    def _camera_dirs(self) -> np.ndarray:
        p = self.player
        vo = self.view_offset
        a, b = math.radians(p.yaw + vo[3]), math.radians(p.pitch + vo[4])
        # カメラ座標 → ワールド: 前 f, 右 r, 上 u
        f = np.array([math.sin(a) * math.cos(b), math.sin(b), math.cos(a) * math.cos(b)])
        r = np.array([-math.cos(a), 0.0, math.sin(a)])
        u = np.cross(r, f)
        cr = self._cam_rays
        d = cr[..., 0:1] * r + cr[..., 1:2] * u + cr[..., 2:3] * f
        return d.reshape(-1, 3)

    def render(self) -> np.ndarray:
        p = self.player
        vo = self.view_offset
        ox, oy, oz = p.x + vo[0], p.y + EYE + vo[1], p.z + vo[2]
        dirs = self._camera_dirs()
        if HAVE_NUMBA:
            ht, hb, hf = _render_nb(self.world, ox, oy, oz, dirs, self.max_dist)
        else:
            ht, hb, hf = _render_py(self.world, ox, oy, oz, dirs, self.max_dist, None, TOP, SIDE)
        hitm = np.isfinite(ht)
        # 空
        up = np.clip(dirs[:, 1], 0, 1)[:, None]
        col = SKY_HORIZON * (1 - up) + SKY_TOP * up
        # ブロック
        pos = np.array([ox, oy, oz]) + dirs * np.where(hitm, ht, 0)[:, None]
        is_top = (hf == 1) & (dirs[:, 1] < 0)
        base = np.where(is_top[:, None], TOP[hb], SIDE[hb])
        shade = np.where(hf == 1, np.where(dirs[:, 1] < 0, 1.0, 0.5), np.where(hf == 0, 0.8, 0.65))
        # 8×8 テクセルのざらつき（運動検出に必要な模様）
        tex = np.floor(pos * 8.0 + 1e-4).astype(np.int64)
        h = (tex[:, 0] * 73856093) ^ (tex[:, 1] * 19349663) ^ (tex[:, 2] * 83492791)
        noise = 0.85 + 0.3 * ((h & 1023) / 1023.0)
        blk = base * (shade * noise)[:, None]
        fog = np.clip(ht / self.max_dist, 0, 1)[:, None] ** 1.5
        blk = blk * (1 - fog) + col * fog  # 遠くはその方向の空の色に溶け込む
        col = np.where(hitm[:, None], blk, col)
        # スライム（簡単な箱のレイ判定）
        depth = np.where(hitm, ht, np.inf)
        o = np.array([ox, oy, oz])
        for s in self.slimes:
            lo = np.array([s.x - s.size / 2, s.y, s.z - s.size / 2])
            hi = lo + s.size
            v = (lo + hi) / 2 - o
            dist = float(np.linalg.norm(v))
            if dist > self.max_dist or dist < 1e-3:
                continue
            # 視線方向の円錐で候補の画素を絞ってから箱との交差判定
            rad = s.size * 0.9 + 0.05
            cosr = math.cos(math.asin(min(1.0, rad / dist))) if dist > rad else -1.0
            cand = np.flatnonzero(dirs @ (v / dist) > cosr)
            if not len(cand):
                continue
            dd = dirs[cand]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (lo - o) / dd
                t2 = (hi - o) / dd
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            ok = (tmax >= np.maximum(tmin, 0)) & (tmin < depth[cand]) & (tmin > 0)
            if not ok.any():
                continue
            m = cand[ok]
            q = o + dirs[m] * tmin[ok, None]
            face = np.argmin(np.abs(np.stack([q - lo, q - hi])).min(0), axis=1)
            sh = np.where(face == 1, 1.0, 0.75)
            c = np.array(s.color, dtype=np.float32) * sh[:, None]
            rel = (q - lo) / s.size  # 目
            eye_m = (rel[:, 1] > 0.55) & (rel[:, 1] < 0.75) & (
                ((np.abs(rel[:, 0] - 0.3) < 0.1) | (np.abs(rel[:, 0] - 0.7) < 0.1))
                | ((np.abs(rel[:, 2] - 0.3) < 0.1) | (np.abs(rel[:, 2] - 0.7) < 0.1)))
            c[eye_m] = 20
            col[m] = c
            depth[m] = tmin[ok]
        img = np.clip(col, 0, 255).astype(np.uint8).reshape(self.height, self.width, 3)
        return img

    def _odor(self):
        """左右の触角で感じる花の匂い（距離の二乗で減衰）。"""
        p = self.player
        r = 10
        x0, x1 = max(0, int(p.x) - r), min(self.size, int(p.x) + r + 1)
        z0, z1 = max(0, int(p.z) - r), min(self.size, int(p.z) + r + 1)
        y0, y1 = max(0, int(p.y) - 3), min(self.world.shape[1], int(p.y) + 4)
        sub = self.world[x0:x1, y0:y1, z0:z1]
        fx, fy, fz = np.nonzero(sub == FLOWER)
        if not len(fx):
            return 0.0, 0.0
        fx = fx + x0 + 0.5
        fz = fz + z0 + 0.5
        out = []
        for side in (+1, -1):  # 左 = +x 側（yaw 0 のとき）
            ax, az = self._forward_vec(self.player.yaw + side * 90)
            px, pz = p.x + 0.25 * ax, p.z + 0.25 * az
            d2 = (fx - px) ** 2 + (fz - pz) ** 2
            out.append(float(np.clip(np.sum(1.0 / (1.0 + d2)) * 0.8, 0, 1)))
        return out[0], out[1]

    def _observe(self) -> Observation:
        p = self.player
        odl, odr = self._odor()
        speed_wind = 0.0
        return Observation(
            frame=self.render(),
            fov_v=self.fov_v,
            sugar=self._sugar,
            bitter=self._bitter,
            touch_left=self._touch[0],
            touch_right=self._touch[1],
            wind=float(np.clip(abs(p.vy) / 20.0 + speed_wind, 0, 1)),
            odor_left=odl,
            odor_right=odr,
            info={"x": round(p.x, 2), "y": round(p.y, 2), "z": round(p.z, 2), "yaw": round(p.yaw, 1),
                  "health": round(p.health, 1), "t": round(self.t, 2)},
        )

    # ------------------------------------------------------------- telemetry
    def minimap(self, cells: int = 96) -> np.ndarray:
        """上から見た地図（ダッシュボード用）。"""
        w = self.world
        size = self.size
        top_idx = (w.shape[1] - 1) - np.argmax((w[:, ::-1, :] != AIR), axis=1)
        top_b = np.take_along_axis(w, top_idx[:, None, :], axis=1)[:, 0, :]
        col = TOP[top_b] * (0.6 + 0.4 * (top_idx / w.shape[1]))[..., None]
        step = max(1, size // cells)
        # [x, z, 3] → 画像の行 = z, 列 = x
        return col[::step, ::step].astype(np.uint8).transpose(1, 0, 2).copy()

    def extra_telemetry(self) -> Dict:
        p = self.player
        return {
            "player": {"x": p.x, "z": p.z, "y": p.y, "yaw": p.yaw, "health": p.health},
            "slimes": [{"x": s.x, "z": s.z} for s in self.slimes],
            "world_size": self.size,
            "events": list(self.events[-8:]),
            "game_stats": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.stats.items()},
        }

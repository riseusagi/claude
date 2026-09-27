# FlyCraft 🪰⛏️ — ハエの脳にマインクラフト統合版をプレイさせる

キイロショウジョウバエの**全脳コネクトーム**（FlyWire：138,639 ニューロン・1,509 万結合・5,449 万シナプス）を、
そのままスパイキングニューラルネットワークとして動かし、

- Minecraft 統合版の画面を**複眼**で見せ、
- 味・接触・風・匂いを**感覚ニューロン**に入れ、
- 脳の**下行性ニューロン**の発火でキャラクターを歩かせる・曲がらせる・跳ばせる・噛ませる

ツールです。行動を決めるのはプログラムされたルールではなく、実際のハエの脳の配線を流れるスパイクです。

![ダッシュボード: ゲーム画面・複眼像・全脳の発火・下行性ニューロンの出力](docs/dashboard.jpg)

---

## できること

| モード | コマンド | 目・体 | 向いている環境 |
|---|---|---|---|
| **内蔵ワールド** | `python -m flycraft sim` | 付属のミニ・ボクセルワールド | まず試す。Minecraft 不要 |
| **統合版 WebSocket** | `python -m flycraft bedrock` | `/connect` で接続し、コマンドで移動 | Windows / スマホ / タブレットの統合版（MOD 不要） |
| **画面 + キー入力** | `python -m flycraft screen` | 画面キャプチャ + 仮想キーボード・マウス | Windows 版の統合版を普通に操作させたいとき |

どのモードでも、ブラウザに**ライブダッシュボード**が開きます:

- ゲーム画面と、ハエの**複眼に写った像**（六角形の個眼）、運動検出の結果
- **全脳 138,639 ニューロンの発火**（FlyWire の座標で正面から見た脳が光る）
- 下行性ニューロンの発火率と、押されているキー（W/A/S/D・Space・クリック）
- **in silico 光遺伝学**：ボタン 1 つで「甘味受容ニューロン」「巨大繊維」「DNa02 左」などを刺激・抑制して、行動が変わる様子を見られる

---

## しくみ

```mermaid
flowchart LR
  G[ゲーム画面] --> E[複眼<br/>3° 間隔の格子]
  E --> L[視葉エミュレーション<br/>HR 運動検出 T4/T5<br/>LPLC2 型ルーミング<br/>小物体検出]
  L --> V[視覚投射ニューロン<br/>約 5,000 個を駆動<br/>受容野はコネクトームから推定]
  S[味・接触・風・匂い・単眼] --> SN[感覚ニューロン]
  V --> B[全脳 LIF モデル<br/>FlyWire v783<br/>138,639 ニューロン]
  SN --> B
  B --> D[下行性ニューロン<br/>P9 / MDN / DNa01・02<br/>巨大繊維 / MN9]
  D --> A[操作<br/>前進・後退・旋回<br/>ジャンプ・噛む]
  A --> G
```

### 脳（`flycraft/brain.py`）

- FlyWire v783 の全ニューロン・全結合をそのまま使う Leaky Integrate-and-Fire モデル。
  パラメータと結合の符号（神経伝達物質の予測）は Shiu et al. 2024 (*Nature*) の全脳モデルと同じ。
- 50 ms ごとに感覚入力を更新して脳を 0.5 ms 刻みで進める。numba で並列化したイベント駆動シミュレーションで、
  4 コアの CPU でほぼ実時間（0.7〜1.2 倍）。
- 検証: 糖受容ニューロンの活性化で口吻伸展の運動ニューロン MN9 が発火する（Shiu et al. の主結果）、
  LPLC2 の活性化で巨大繊維が発火する、などをテストで確認している。

### 目（`flycraft/vision.py`, `flycraft/retinotopy.py`）

1. 画面を 3° 間隔の格子に平均化して「個眼」の輝度を得る（本物のハエは約 5°）。
2. **視葉をソフトウェアで模倣**:
   ハッセンシュタイン・ライヒャルト型の運動検出器（T4/T5 相当）で局所運動を計算し、
   - LPLC2 / LC4 など: 受容野の**上下左右 4 方向すべてで外向き**の運動（迫ってくる物体）にだけ応答（Klapoetke et al. 2017）。
     歩いたときの地面の流れや旋回では応答しない
   - LC10 / LC11 / LC17 など: 周囲と違う動きをする小さな物体
   - HS / LLPC: 前→後の運動、LPC / H2: 後→前、VS: 下向き、MeTu: 明るさ
3. 各視覚投射ニューロンが**視野のどこを見ているか**はコネクトームから推定する:
   視細胞の終末位置の主成分分析で視野の向きを決め（複眼背縁 DRA が上を向くことで検証）、
   シナプスをたどって下流へ伝播させる。

### 体（`flycraft/motor.py`, `flycraft/senses.py`）

| 行動 | 読み出すニューロン | 根拠 |
|---|---|---|
| 前進（W） | DNp09 (P9), DNg100 (BDN2) | Bidaye et al. 2020 |
| 後退（S） | MDN（ムーンウォーカー） | Bidaye et al. 2014 |
| 旋回（マウス左右） | DNa02, DNa01 の左右差（同側へ曲がる） | Rayshubskiy et al. 2020 (bioRxiv) |
| 逃避ジャンプ（Space） | DNp01（巨大繊維）, DNp02/04/11 | von Reyn et al. 2014, Ache et al. 2019 |
| 噛む（左クリック） | MN9（口吻伸展運動ニューロン） | Shiu et al. 2024 |

| 感覚 | 入力するニューロン |
|---|---|
| 甘味（花を舐めた・アイテムを拾った） | 糖受容ニューロン 129 個 |
| 苦味・痛み（サボテン・ダメージ） | 苦味受容ニューロン 65 個 |
| 接触（壁にぶつかった、左右別） | 頭部・複眼の機械感覚剛毛 |
| 風（落下・移動） | ジョンストン器官の風・重力ニューロン |
| 匂い（近くの花、左右の触角） | 食べ物の匂いの嗅覚受容ニューロン |
| 単眼 | 画面上部の明暗変化 |

### どこまでが「本物」か（大事な注意）

- **本物**: 配線（全 1,509 万結合とその符号）、細胞タイプの注釈、LIF の定数。行動を決めているのはこの配線上のスパイク。
- **足したもの・簡略化したもの**:
  - **短期シナプス抑圧**: 元のモデルに持続的な感覚入力を与え続けると、触角葉の興奮性ループから全脳が 400 Hz で発火し続ける「てんかん様の暴走」に落ちることを確認したため追加（`--no-std` で元のモデル）。
  - **視葉はソフトウェアで模倣**: 視細胞を直接駆動しても、信号は視葉の 2〜3 シナプス先で消えて行動に届かないことを確認した（視葉の計算は非スパイクの段階的電位に依存するため）。そこで視葉の既知の機能を模倣し、中枢脳の入口（視覚投射ニューロン）から先をコネクトームに任せている。
  - **空腹ドライブ**: 空腹度に応じて前進指令ニューロン P9 に定常電流を入れる（実際のハエも空腹で歩き回る）。花の蜜で満たされる。
  - **胸部神経節の反射**: FlyWire の「脳」データには歩行の回路がある胸部神経節が含まれないため、壁を押し続けたら向きを変える単純な反射だけ規則で入れている（`--no-reflex` で無効）。
  - 学習（シナプス可塑性）は無い。毎回同じ配線で反応する。
- 行動はきれいではありません。ハエの脳にとって Minecraft は異世界で、報酬で訓練もしていないので、**迷い・ぶつかり・跳ね回ります**。それがこのツールの見どころです。

---

## 必要なもの

- Python 3.9 以上（Windows / macOS / Linux）
- CPU 4 コア以上推奨、メモリ 4 GB 以上（初回のデータ変換時）
- ディスク約 200 MB（コネクトームのデータ）

## インストール

```bash
git clone https://github.com/riseusagi/claude.git flycraft
cd flycraft
pip install -e ".[all]"        # numba（高速化）と mss（画面キャプチャ）も入る
python -m flycraft download    # FlyWire のデータを取得（約 135 MB、初回のみ）
```

`numba` が無くても動きますが、数倍〜10 倍遅くなります。

## まずは内蔵ワールドで

```bash
python -m flycraft sim
```

ブラウザでダッシュボード（http://localhost:8765/）が開き、ハエが内蔵の草原を歩き回ります。
花（甘い）、サボテン（苦い・痛い）、跳ねるスライム（迫ってくる物体）があります。

データをダウンロードせずに雰囲気だけ見たいときは `--toy`（人工の小さな回路。本物の脳ではありません）:

```bash
python -m flycraft sim --toy
```

---

## Minecraft 統合版で遊ばせる

> ⚠ ハエは地形を壊したり（噛む）、崖から落ちたりします。**テスト用のワールド**で試してください。

### 方法 A: WebSocket（`/connect`）— MOD 不要・おすすめ

1. `python -m flycraft bedrock` を実行（ポート 19131 で待ち受け）
2. 統合版の **設定 → 一般 →「暗号化された WebSocket を要求」をオフ**
3. **チートを有効**にしたワールドに入り、チャットで

   ```
   /connect localhost:19131
   ```

   スマホやタブレットの統合版からは `localhost` の代わりに PC の IP アドレスを指定します。

4. 「✅ Minecraft が接続しました」と出たら、ハエが体を動かし始めます。

- 移動は `tp` の相対移動、ジャンプは上向きの `tp`、噛むは視線の先のブロックを壊す（`setblock ... destroy`）・近くの生き物に `damage`。
- 目は、既定では `execute ... if block` の光線プローブで作る粗い奥行き画像です。
  **同じ PC で遊んでいるなら `--screen-vision`** を付けると、画面キャプチャの本物の映像で見ます:

  ```bash
  python -m flycraft bedrock --screen-vision
  ```
- 足元の花・甘いベリー・蜂蜜ブロックで甘味、サボテン・マグマ・火で苦味、アイテムを拾うと甘味、死ぬと苦味。
- チャットで `!fly stop` / `!fly go` / `!fly hunger 0.9` と打つと、止める・再開・空腹度の変更ができます。

<details>
<summary>接続できないとき</summary>

- 「暗号化された WebSocket を要求」がオフか確認。
- Windows で `localhost` に接続できない場合（古い UWP 版の制限）、管理者の PowerShell で次を実行:
  `CheckNetIsolation.exe LoopbackExempt -a -n="Microsoft.MinecraftUWP_8wekyb3d8bbwe"`
- ファイアウォールでポート 19131 を許可（別の端末から接続する場合）。
- ポートを変えるには `--ws-port 20000`（そのときは `/connect localhost:20000`）。

</details>

### 方法 B: 画面を見てキーボード・マウスで操作（Windows）

```bash
python -m flycraft screen
```

- Minecraft のウィンドウをキャプチャし、W / S / Space / 左クリックとマウス移動を**本物の入力として送ります**。
- **Minecraft のウィンドウが最前面のときだけ**入力を送ります。**F8 で一時停止／再開**、ターミナルで Ctrl+C で終了。
- おすすめ設定: 統合版の「**自動ジャンプ**」をオン、ウィンドウモード、視野角は `--fov` に合わせる（既定 70）。
- 旋回が速すぎ／遅すぎるときは `--mouse-speed`（既定 6）で調整。
- macOS / Linux では `pip install pynput` で動く場合があります（キャプチャ対象は `--region x,y,w,h` で指定）。

---

## in silico 実験（光遺伝学ごっこ）

ダッシュボードのボタンで、ニューロン群を刺激したときの行動の変化を見られます。
コマンドラインでは、刺激に対する下行性ニューロンの応答を直接調べられます:

```bash
python -m flycraft probe sugar                 # 糖受容ニューロン → MN9（口吻伸展）が発火
python -m flycraft probe LPLC2 --side left     # ルーミング検出 → 巨大繊維 DNp01
python -m flycraft probe LC10a --side left     # 小物体検出 → 同側の DNa02（そちらへ曲がる）
python -m flycraft probe LC9 --side left       # → P9（前進）
```

細胞タイプ名は FlyWire の注釈（[Codex](https://codex.flywire.ai/) で検索可）をそのまま使えます。

---

## 主なオプション

| オプション | 意味 |
|---|---|
| `--hunger 0.8` | 初期の空腹度（高いほどよく歩く） |
| `--acuity 5` | 複眼の解像度（度）。5 で本物のハエ並み |
| `--fov 70` | ゲームの視野角（垂直） |
| `--dt 1.0` | 積分ステップ。粗くなるが約 2 倍速い |
| `--threads 8` | numba のスレッド数 |
| `--no-std` | 短期シナプス抑圧なし（Shiu et al. の元のモデル。暴走しやすい） |
| `--no-reflex` | 胸部神経節の障害物反射を切る（純粋に脳だけで動く） |
| `--retina` | 視細胞も直接駆動する（行動への影響はほぼ無いがダッシュボードの視葉が光る） |
| `--seconds 60` | 脳の時間で 60 秒動かして終了 |
| `--port 8765` / `--no-browser` / `--no-dashboard` | ダッシュボードの設定 |
| `--toy` | 人工の小さな回路で動かす（データ不要） |

`python -m flycraft bench` で、脳のシミュレーション速度を測れます。

---

## データ・引用

データはこのリポジトリに含めず、初回に配布元から直接ダウンロードします（`~/.cache/flycraft`、
Windows は `%LOCALAPPDATA%\flycraft`。環境変数 `FLYCRAFT_DATA` で変更可）。
データの利用条件は各配布元に従ってください。研究などで使う場合は以下を引用してください。

- **FlyWire コネクトーム**: Dorkenwald et al. (2024) Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138.
- **細胞注釈**: Schlegel et al. (2024) Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634, 139–152.
- **全脳 LIF モデルと派生データ**: Shiu et al. (2024) A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634, 210–219.
- **神経伝達物質の予測**: Eckstein et al. (2024) Neurotransmitter classification from electron microscopy images at synaptic sites in *Drosophila melanogaster*. *Cell*.
- 行動とニューロンの対応: Bidaye et al. 2014 (*Science*), Bidaye et al. 2020 (*Neuron*), von Reyn et al. 2014 (*Nat Neurosci*), Ache et al. 2019 (*Curr Biol*), Klapoetke et al. 2017 (*Nature*), Rayshubskiy et al. 2020 (*bioRxiv*)

Minecraft は Mojang Studios / Microsoft の商標です。本ツールは非公式のもので、Mojang・Microsoft・FlyWire とは関係ありません。

---

## 開発

```bash
pip install -e ".[all,dev]"
pytest                      # FlyWire のデータがあれば、実データの検証テストも走る
```

| ファイル | 役割 |
|---|---|
| `flycraft/brain.py` | 全脳 LIF シミュレータ（numpy / numba） |
| `flycraft/connectome.py`, `data.py` | コネクトームの構造・ダウンロード・キャッシュ |
| `flycraft/retinotopy.py` | コネクトームから網膜位相（視野上の位置）を推定 |
| `flycraft/vision.py` | 複眼・視葉エミュレーション・視覚投射ニューロンの駆動 |
| `flycraft/senses.py`, `motor.py` | 感覚ニューロンへの入力、下行性ニューロンからの読み出し、反射 |
| `flycraft/fly.py`, `runner.py` | ハエ本体と実行ループ |
| `flycraft/backends/` | `sim`（内蔵ワールド）, `bedrock_ws`（/connect）, `screen`（画面 + キー入力） |
| `flycraft/dashboard/` | ダッシュボード（標準ライブラリの HTTP サーバー + 単一 HTML） |
| `flycraft/toy.py` | デモ・テスト用の人工回路 |
| `tests/mock_bedrock.py` | 統合版の WebSocket プロトコルを真似る疑似クライアント（テスト用） |

"""BUGC2 本番コントローラ（カメラ → 経路計画 → 実機へ走行指令）

実機API:  GET http://<ROBOT_IP>/cmd?k=<w|a|s|d>&s=<0-100>
          例) curl "http://192.168.100.250/cmd?k=w&s=40"

初回のみ Anaconda Prompt:
  conda create -n bugc2 python=3.11 -y
  conda activate bugc2
  pip install opencv-contrib-python numpy

実行例:
  # 会場のカメラ + 実機
  python test.py --camera http://192.168.100.24/video_feed --robot http://192.168.100.250
  # PC内蔵カメラで練習
  python test.py --camera 0 --robot http://192.168.100.250
  # 実機に送らずに動きだけ確認
  python test.py --camera 0 --dry-run
  #ルールはURL記載
  https://sites.google.com/view/cq-ws/

キー操作（映像ウィンドウをアクティブにして押す）
  Enter … 完全自律スタート（ゴールするまで自分で走り続ける）
  Space … 停止（自動走行も解除）
  W A S D … マニュアル操作（前進 / 左旋回 / 後進 / 右旋回）
  + / -  … 走行速度 ±5     [ / ]  … 旋回速度 ±5
  1〜9   … 走行速度をその値×10 に（4→40）   0 … 旋回速度を走行速度と同じに戻す

旋回が回り過ぎるとき
  遅れのぶん先読みして早めに止め、さらに旋回を短いパルスに区切っている。
  それでも回り過ぎるなら:
    --turn-speed 20     旋回を遅くする（一番効く）
    --turn-pulse 0.10   1回の旋回を短くする
    --turn-lead 0.35    もっと手前で止める
  R      … キャリブレーションと記憶をやり直す
  P      … キー割り当ての確認（w→a→s→d を 0.4 秒ずつ送る）
  Q / Esc … 停止して終了

表示が見切れるとき
  ウィンドウは自由にリサイズできます（WINDOW_NORMAL）。
  既定では表示だけ 1280x720 以内に縮小します（検出は元解像度のまま）。
    --view 1600x900 … 上限を変える
    --view 0        … 縮小しない（等倍）

マーカID（公式ルール https://sites.google.com/view/cq-ws/）
  0=ゴール / 1〜9=プレイヤー / 10〜19=お邪魔ロボット
  20〜23=フィールド四隅（これが4つ写ると自動キャリブ）/ 30〜=壁（偶数→+1奇数が1枚）

安全のしくみ
  * 自機マーカを一時的に見失っても BLIND_S=0.6 秒は指令から位置を推測して走り続ける。
    それより長く見えないときは止まって待ち、見えたら自動で再開する（諦めない）
  * 相手・妨害は見失っても HOLD_S=0.8 秒は居るものとして扱う（点滅で突っ込まない）
  * 物理的にあり得ない位置の飛び（誤検出・ID混同）は棄却する。ただし
    本当に動かされた場合は数フレームで受け入れる
  * 映像が切れたら終了せずに開き直して自律を続ける
  * 指令には有効期限 CMD_TTL=0.35s がある。制御ループが止まっても
    送信スレッドが自動で停止指令に切り替える（暴走防止）
  * 画面右下に「実際に送っている指令」を大きく出す。STOP と出ていれば
    停止指令を送っている状態

走らせ方の工夫
  * 速度は低速（走行40 / 旋回25）が既定。s を小さくしすぎると摩擦に負けて
    「前進指令を出しているのに動かない」が起きるため。
    走りながら +/- や数字キーで変えられる（--speed / --turn-speed でも指定可）
  * 「マーカの向き」と「実際に進む向き」のズレを走りながら学習して補正する。
    マーカを斜めに貼っても、駆動に癖があっても自動で合う（画面の 向き補正 に出る）
  * 前進指令なのに動いていない状態が 1.2 秒続いたら、後退して向きを変える
  * 相手(1〜9のうち自分以外)と妨害(10〜19)は経路では避けない。前方に来たら
    止まって待ち、1.5秒どかなければ横へ迂回する（遠回りせずに最短を狙う）

大事な前提
  * BUGC2 に貼るマーカは、**上辺が進行方向**を向くように貼る
  * マーカが機体の中心でなく後方に付いている場合は、マーカ中心から
    回転中心（車軸あたり）までの距離を測って --marker-back に入れる。
    入れないとゴール判定が遅れ、旋回のたびに狙いがぶれる
    （マーカの「上向き」ではなく「左上→右上」の向きが前）
  * ゴールと壁のマーカは一度見えたら記憶する。自機がゴールマーカに
    乗ると隠れてしまい、到達判定ができなくなるため
  * 実機のキー割り当ては先頭の KEY_* 定数で変更できる（P キーで確認）

動作確認: 模擬カメラ（MJPEG・四隅マーカ・壁2枚）と模擬機体で通しテスト済み。
          自動キャリブ → 経路計画 → w/a/s/d 指令 → 12.5 秒でゴール、送信失敗 0。
"""
from __future__ import annotations

import argparse
import heapq
import math
import queue
import random
import threading
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import cv2
import numpy as np

# ---------------------------------------------------------------- 設定
VERSION = "rev3 2026-08-21"           # 起動時に表示。古いファイルを掴んでいないかの確認用

FIELD_W, FIELD_H = 1000.0, 500.0     # 四隅マーカ中心の間隔 [mm]
ROBOT_R, MARGIN, WALL_HALF = 40.0, 18.0, 8.0
# マーカが機体の中心ではなく後方に付いている場合の補正[mm]。
# 「マーカ中心 → 機体の回転中心（車軸のあたり）」までの距離を前向きに測って入れる。
# 実測して --marker-back で指定するのが確実。
MARKER_BACK = 45.0
GOAL_R = 45.0                        # ゴール判定半径 [mm]

# 実機のキー割り当て。違っていたらここだけ直す（P キーで確認できる）
KEY_FWD, KEY_BACK, KEY_LEFT, KEY_RIGHT = "w", "s", "a", "d"
# 停止のために送る指令の並び。実機が s=0 で止まらない場合はここに足す
#   例) [("w", 0), ("x", 0)]
STOP_CMDS = [("w", 0)]

SEND_HZ = 15.0                      # 実機へ送る頻度
CMD_TTL = 0.35                      # 指令の有効期限[s]。これを過ぎたら勝手に停止を送る
                                    #   （映像が固まる・PC側が止まる等で暴走させないため）
# 完全自律のための追跡パラメータ
BLIND_S = 0.6                       # 自機マーカを見失っても、この時間は推定で走り続ける[s]
                                    #   これを超えたら停止して待つ（諦めずに復帰したら再開）
HOLD_S = 0.8                        # 他機を見失っても、この時間は居るものとして扱う[s]
JUMP_MAX_V = 600.0                  # 位置の飛びを棄却する上限速度[mm/s]
JUMP_MARGIN = 45.0                  # 飛び判定の余裕[mm]（検出ノイズ分）
JUMP_ACCEPT = 4                     # 連続でこの回数はじかれたら「本当に動いた」と受け入れる
STATIC_MOVE = 60.0                  # 静止物(ゴール/壁)がこれ以上動いたら置き直しと見なす[mm]

FIXED_SPEED = 60                    # 走行速度（前進・後進）。低速を既定に。走りながら変更できる
TURN_SPEED = 60                     # 旋回速度。回る量はパルス時間で決めるので走行と同じでよい

# 旋回の回り過ぎ対策。
# カメラ〜検出〜送信で 0.2 秒前後の遅れがあるため、
# 「誤差が消えるまで回し続ける」と見えた時点ではもう回り過ぎている。
# そこで ①遅れぶんを先読みして早めに止め ②旋回を短いパルスに区切って毎回測り直す。
TURN_LEAD_S = 0.0                   # 先読み[s]。短いパルスで測り直すので既定は0
TURN_PULSE = 0.05                   # 旋回1パルスの長さ[s]（実機実測: 0.05s で約40度）
TURN_DEG_PER_PULSE = 40.0           # TURN_PULSE 1回で回る角度[deg]。実機を見て直す
TURN_GAP = 0.40                     # パルス後に止まって測り直す時間[s]。カメラ遅延より長くする
                                    #   （短いと前のパルスの結果を見ずに次を打って振動する）
TURN_LEAD_MAX = 20.0                # 先読みの上限[deg]（これ以上手前で止めない）
TURN_SAFETY = 1.15                  # 旋回速度の見積りを安全側（速め）に振る係数
TURN_IN = 22.0                      # このずれ[deg]以上で旋回に入る
TURN_OUT = 12.0                     # ここまで合ったら前進に移る（ヒステリシス）
LEARN_GAIN = 0.20                   # 「マーカの向き」と「実際に進む向き」のずれの学習係数
CAL_PULSE = 0.25                    # 向き合わせの1回の前進時間[s]（短く刻んで安全に測る）
CAL_SETTLE = 0.20                   # パルス後に止まって測る時間[s]
CAL_TRIES = 6                       # 向き合わせの最大回数
MOVE_EPS = 6.0                      # これ以上動いたら「動いた」と見なす[mm]

# 他機の扱い。公式はトーナメントの1vs1なので、相手は 1〜9 のうち自分以外の1機。
# 妨害ロボットは 10〜19。壁と違って動くので経路では避けず、
# 目の前に来たときだけ待つ／よける（完全回避を狙うと遠回りで負ける）
AVOID_R = 130.0                     # この距離[mm]以内を「近い」と見なす
AVOID_FOV = 55.0                    # 進行方向から±この角度[deg]以内なら「前方」
AVOID_WAIT = 1.5                    # これ以上待たされたら迂回に切り替える[s]
ROBOT_IDS = range(1, 10)            # プレイヤー（自機・ライバル）
JAM_IDS = range(10, 20)             # 妨害ロボット
REPLAN_DT = 0.4                     # 経路の再計算周期[s]


# ---------------------------------------------------------------- 幾何
def norm180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def point_seg_dist(p, a, b) -> float:
    p, a, b = map(np.asarray, (p, a, b))
    d = b - a
    q = float(d @ d)
    if q < 1e-9:
        return dist(p, a)
    t = float(np.clip(((p - a) @ d) / q, 0.0, 1.0))
    return dist(p, a + d * t)


def _cross(a, b, c) -> float:
    a, b, c = map(np.asarray, (a, b, c))
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def seg_cross(a, b, c, d) -> bool:
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def seg_seg_dist(a, b, c, d) -> float:
    if seg_cross(a, b, c, d):
        return 0.0
    return min(point_seg_dist(a, c, d), point_seg_dist(b, c, d),
               point_seg_dist(c, a, b), point_seg_dist(d, a, b))


# ---------------------------------------------------------------- 実機への送信
class Robot:
    """走行指令を別スレッドで送る。

    HTTP を制御ループの中で直接呼ぶと、1回のタイムアウトで映像が固まる。
    最新の指令だけを保持して、専用スレッドから一定間隔で投げ続ける。

    安全のための仕掛け（ここが無いと「映っていないのに走り続ける」が起きる）
      * 指令に有効期限 CMD_TTL を付ける。制御ループが指令を出し続けないと
        自動で停止指令に切り替わる。映像が固まっても暴走しない
      * stop() は待たずに即送信し、続けて数回送る
    """

    def __init__(self, base_url: str, dry_run: bool = False):
        self.base = base_url.rstrip("/")
        self.dry = dry_run
        self.cmd = (STOP_CMDS[0][0], 0)      # これから送る指令
        self.cmd_t = 0.0                     # cmd がセットされた時刻
        self.cur = (STOP_CMDS[0][0], 0)      # 実際に最後に送った指令
        self.expired = False                 # 有効期限切れで止めているか
        self.stop_burst = 0                   # 停止指令を続けて送る残り回数
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.alive = True
        self.sent = 0
        self.errors = 0
        self.last_err = ""
        self.last_url = ""
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def set(self, key: str, speed: int) -> None:
        speed = int(max(0, min(100, speed)))
        with self.lock:
            self.cmd = (key, speed)
            self.cmd_t = time.monotonic()
            self.expired = False
        self.wake.set()                      # 100ms 待たずにすぐ送る

    def stop(self) -> None:
        with self.lock:
            self.cmd = (STOP_CMDS[0][0], 0)
            self.cmd_t = time.monotonic()
            self.stop_burst = max(self.stop_burst, len(STOP_CMDS) + 1)
        self.wake.set()

    def _send(self, key: str, speed: int) -> None:
        url = f"{self.base}/cmd?{urlencode({'k': key, 's': speed})}"
        self.last_url = url
        self.cur = (key, speed)
        if self.dry:
            self.sent += 1
            return
        try:
            with urlopen(url, timeout=0.35) as r:
                r.read()
            self.sent += 1
        except Exception as e:                      # noqa: BLE001  通信断は想定内
            self.errors += 1
            self.last_err = f"{type(e).__name__}: {e}"

    def _loop(self) -> None:
        interval = 1.0 / SEND_HZ
        while self.alive:
            try:
                now = time.monotonic()
                with self.lock:
                    key, speed = self.cmd
                    age = now - self.cmd_t
                    burst = self.stop_burst
                    if burst > 0:
                        self.stop_burst = burst - 1
                if burst > 0:
                    # 停止は取りこぼしたくないので、並びを続けて送る
                    for k, sp in STOP_CMDS:
                        self._send(k, sp)
                    self.wake.wait(0.03)
                    self.wake.clear()
                    continue
                if speed != 0 and age > CMD_TTL:
                    # 制御ループが指令を更新できていない → 止める
                    with self.lock:
                        self.expired = True
                        self.cmd = (STOP_CMDS[0][0], 0)
                    for k, sp in STOP_CMDS:
                        self._send(k, sp)
                else:
                    self._send(key, speed)
            except Exception as e:                   # noqa: BLE001  ここで死なせない
                self.errors += 1
                self.last_err = f"{type(e).__name__}: {e}"
            self.wake.wait(interval)
            self.wake.clear()

    def close(self) -> None:
        # 先に送信スレッドを止めてから停止を送る。
        # 逆順にすると、停止の後に古い指令がもう一度届いて走り続けることがある。
        with self.lock:
            self.cmd = (STOP_CMDS[0][0], 0)
            self.cmd_t = time.monotonic()
            self.stop_burst = 0
        self.alive = False
        self.wake.set()
        self.th.join(timeout=1.5)
        for _ in range(3):                           # 終了時は確実に止める
            for k, sp in STOP_CMDS:
                self._send(k, sp)
            time.sleep(0.05)

    def probe(self) -> None:
        """キー割り当ての確認。w→a→s→d を 0.4 秒ずつ動かして、最後に停止。"""
        for k, name in ((KEY_FWD, "前進"), (KEY_LEFT, "左旋回"),
                        (KEY_BACK, "後進"), (KEY_RIGHT, "右旋回")):
            print(f"  k={k} ({name}) を 0.4 秒…")
            self.set(k, 35)
            time.sleep(0.4)
            self.stop()
            time.sleep(0.35)
        print("  停止も効いているか確認してください。")
        print("  動きと合っていなければ先頭の KEY_* を、止まらなければ STOP_CMDS を直します。")


def local_ip() -> str:
    """このPCがLANで使っているIPを調べる（外に接続はしない）。"""
    import socket
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sk.connect(("192.168.1.1", 1))
        return sk.getsockname()[0]
    except Exception:                                # noqa: BLE001
        return "127.0.0.1"
    finally:
        sk.close()


def check_robot(base: str, timeout: float = 2.0) -> tuple[bool, str]:
    """機体に1回だけ停止指令を送って、届くかどうかを確かめる。"""
    k, sp = STOP_CMDS[0]
    url = f"{base.rstrip('/')}/cmd?{urlencode({'k': k, 's': sp})}"
    t0 = time.monotonic()
    try:
        with urlopen(url, timeout=timeout) as r:
            body = r.read(80).decode("utf-8", "replace").strip()
        return True, f"OK {(time.monotonic()-t0)*1000:.0f}ms  {url}  応答={body!r}"
    except Exception as e:                           # noqa: BLE001
        return False, f"NG {type(e).__name__}: {e}\n     宛先: {url}"


def scan_robots(base24: str | None = None, port: int = 80, timeout: float = 0.6) -> list[str]:
    """同じネットワークを総当たりして、/cmd に応答する機体を探す。"""
    import concurrent.futures as cf
    if base24 is None:
        base24 = ".".join(local_ip().split(".")[:3])
    print(f"{base24}.1〜254 を探索中（port {port}）…")
    hosts = [f"{base24}.{i}" for i in range(1, 255)]
    k, sp = STOP_CMDS[0]

    def probe(h: str):
        u = f"http://{h}:{port}/cmd?{urlencode({'k': k, 's': sp})}"
        try:
            with urlopen(u, timeout=timeout) as r:
                r.read(40)
            return h
        except Exception:                            # noqa: BLE001
            return None

    found = []
    with cf.ThreadPoolExecutor(max_workers=128) as ex:
        for h in ex.map(probe, hosts):
            if h:
                found.append(h)
    return found


# ---------------------------------------------------------------- 視覚
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
try:
    _DETECTOR = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
except AttributeError:                              # 古い OpenCV
    _DETECTOR = None


def detect(gray):
    if _DETECTOR is not None:
        corners, ids, _ = _DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT)
    out = {}
    if ids is not None:
        for i, c in zip(ids.flatten(), corners):
            q = c.reshape(4, 2).astype(np.float32)
            out[int(i)] = q
    return out


def calibrate(marks):
    """四隅マーカ 20〜23 から 画像px → mm の射影変換を作る。

    ID の並び順には依存させず、幾何的に 左上→右上→右下→左下 に並べ替える。
    """
    pts = [marks[i].mean(axis=0) for i in range(20, 24) if i in marks]
    if len(pts) != 4:
        return None, None
    pts = np.float32(pts)
    s, d = pts.sum(axis=1), (pts[:, 0] - pts[:, 1])
    tl, br = pts[int(np.argmin(s))], pts[int(np.argmax(s))]
    bl, tr = pts[int(np.argmin(d))], pts[int(np.argmax(d))]
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [FIELD_W, 0], [FIELD_W, FIELD_H], [0, FIELD_H]])
    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)


def to_field(H, p):
    q = cv2.perspectiveTransform(np.float32([[p]]), H)[0, 0]
    return np.array([float(q[0]), float(q[1])])


def to_image(Hi, p):
    q = cv2.perspectiveTransform(np.float32([[p]]), Hi)[0, 0]
    return int(round(float(q[0]))), int(round(float(q[1])))


def control_point(pose, head_eff, back=None):
    """マーカ位置から、機体の回転中心（制御に使う点）を求める。

    マーカが後方に付いていると、マーカの位置で制御すると
    ・ゴール判定が実機より遅れる
    ・旋回時にマーカが振れて狙いがぶれる
    ので、向きの分だけ前へずらした点を使う。
    """
    b = MARKER_BACK if back is None else back
    if abs(b) < 1e-6:
        return np.array(pose, float)
    r = math.radians(head_eff)
    return np.array(pose, float) + np.array([math.cos(r), math.sin(r)]) * b


def pose_of(H, quad):
    """マーカ四隅から mm 座標の中心と向き（度・時計回り正）を出す。

    OpenCV の corners は マーカ座標系の 左上→右上→右下→左下 の順。
    向きは「左上→右上」= マーカの上辺が指す方向を前方とする。
    （中心→上辺の中点 を使うと 90 度ずれるので注意）

    実機では BUGC2 の進行方向に、マーカの上辺の向きを合わせて貼る。
    """
    c = to_field(H, quad.mean(axis=0))
    tl = to_field(H, quad[0])
    tr = to_field(H, quad[1])
    return c, math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))


def others_from(marks_mm, my_id):
    """他機（ライバル 1〜9 の自分以外 ＋ 妨害 10〜19）の mm 位置。

    公式は 1vs1 なので通常ここに入るのは「相手1機 + 妨害」だけになる。
    """
    out = []
    for i, p in marks_mm.items():
        if i == my_id:
            continue
        if i in ROBOT_IDS or i in JAM_IDS:
            out.append((i, p))
    return out


def walls_from(marks_mm):
    """ID30〜: 若い偶数 → +1 の奇数 を結ぶ直線が1枚の壁。"""
    out = []
    for e in range(30, 97, 2):
        if e in marks_mm and e + 1 in marks_mm:
            out.append((marks_mm[e], marks_mm[e + 1]))
    return out


# ---------------------------------------------------------------- 追跡
class Track:
    """1つのマーカの位置・向き・速度を持ち続ける入れ物。"""

    def __init__(self, pos, deg, t):
        self.pos = np.array(pos, float)
        self.deg = float(deg)
        self.t = t
        self.vel = np.zeros(2)
        self.seen_t = t
        self.rej = 0
        self.n = 1

    def predict(self, now):
        """最後に見えた時点から等速で進んだと仮定した位置。"""
        dt = max(0.0, now - self.t)
        return self.pos + self.vel * min(dt, 0.6)


class Tracker:
    """マーカを見失っても短時間は保持し、あり得ない飛びは棄却する。

    実機では
      * マーカが一瞬写らない（手前を他機が通る・ブレる・逆光）
      * 別IDを誤検出して位置が飛ぶ
      * 実際に誰かが機体を持ち上げて移動させる
    が普通に起きる。これを1か所で吸収して、上の制御は素直に書けるようにする。
    """

    def __init__(self):
        self.tr: dict[int, Track] = {}

    def update(self, now, mid, pos, deg):
        t = self.tr.get(mid)
        if t is None:
            self.tr[mid] = Track(pos, deg, now)
            return True
        dt = max(1e-3, now - t.t)
        moved = dist(pos, t.pos)
        limit = JUMP_MAX_V * dt + JUMP_MARGIN
        if moved > limit and t.rej < JUMP_ACCEPT:
            t.rej += 1                       # あり得ない飛び → 1回は疑う
            return False
        t.rej = 0
        nv = (np.array(pos, float) - t.pos) / dt
        t.vel = t.vel * 0.6 + nv * 0.4 if t.n > 1 else nv
        t.pos = np.array(pos, float)
        t.deg = float(deg)
        t.t = now
        t.seen_t = now
        t.n += 1
        return True

    def get(self, now, mid, hold=HOLD_S):
        """(位置, 向き, 何秒前に見えたか, 推定かどうか) / 期限切れなら None"""
        t = self.tr.get(mid)
        if t is None:
            return None
        age = now - t.seen_t
        if age > hold:
            return None
        if age <= 1e-6:
            return t.pos.copy(), t.deg, 0.0, False
        return t.predict(now), t.deg, age, True

    def alive(self, now, ids, hold=HOLD_S):
        out = []
        for i in ids:
            g = self.get(now, i, hold)
            if g is not None:
                out.append((i, g[0]))
        return out

    def drop(self, mid):
        self.tr.pop(mid, None)


# ---------------------------------------------------------------- 経路計画
class Planner:
    """可視グラフ + ダイクストラ。壁だけを通行禁止にして最短を引く。

    他機は「通れない」にしない（塞がれて経路なしになるのを防ぐ）。
    ぶつかりそうな瞬間は下の走行制御で減速・後退して避ける。
    """

    def __init__(self):
        self.path: list[np.ndarray] = []
        self.idx = 0
        self.plan_t = -99.0

    def free(self, a, b, walls, pad: float | None = None) -> bool:
        edge = (ROBOT_R + MARGIN) if pad is None else pad
        if min(a[0], b[0]) < edge or min(a[1], b[1]) < edge:
            return False
        if max(a[0], b[0]) > FIELD_W - edge or max(a[1], b[1]) > FIELD_H - edge:
            return False
        return all(seg_seg_dist(a, b, w0, w1) >= edge + WALL_HALF for w0, w1 in walls)

    @staticmethod
    def clearance(p, walls) -> float:
        """点 p の、壁と場外までの余裕[mm]（機体半径を引いた実効値）。"""
        c = min(p[0], p[1], FIELD_W - p[0], FIELD_H - p[1]) - ROBOT_R
        for w0, w1 in walls:
            c = min(c, point_seg_dist(p, w0, w1) - WALL_HALF - ROBOT_R)
        return c

    @staticmethod
    def escape_dir(p, walls):
        """一番近い障害物・壁から離れる向き（単位ベクトル）。"""
        best, bd = None, math.inf
        cands = [(np.array([ROBOT_R + MARGIN, p[1]]), p[0]),
                 (np.array([FIELD_W - ROBOT_R - MARGIN, p[1]]), FIELD_W - p[0]),
                 (np.array([p[0], ROBOT_R + MARGIN]), p[1]),
                 (np.array([p[0], FIELD_H - ROBOT_R - MARGIN]), FIELD_H - p[1])]
        for q, d in cands:
            if d < bd:
                bd, best = d, q
        for w0, w1 in walls:
            w0, w1 = np.asarray(w0, float), np.asarray(w1, float)
            d = w1 - w0
            q = float(d @ d)
            t = 0.0 if q < 1e-9 else float(np.clip(((p - w0) @ d) / q, 0.0, 1.0))
            foot = w0 + d * t
            dd = dist(p, foot)
            if dd < bd:
                bd, best = dd, foot
        if best is None:
            return np.array([1.0, 0.0])
        v = np.asarray(p, float) - np.asarray(best, float)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([1.0, 0.0])

    def plan(self, start, goal, walls):
        # 壁に寄りすぎている状態から抜け出せるよう、始点から出る辺だけ余裕を緩める。
        # これが無いと、余裕(58mm)より内側に入った瞬間に「経路なし」で詰む。
        pad0 = min(ROBOT_R + MARGIN, max(ROBOT_R + 3.0, self.clearance(start, walls) + ROBOT_R - 2.0))
        if self.free(start, goal, walls, pad0):
            return [np.array(start, float), np.array(goal, float)]
        pad = ROBOT_R + MARGIN + WALL_HALF + 5.0
        nodes = [np.array(start, float), np.array(goal, float)]
        lo, hi_x, hi_y = ROBOT_R + MARGIN, FIELD_W - ROBOT_R - MARGIN, FIELD_H - ROBOT_R - MARGIN
        for a, b in walls:                            # 壁の端点の外側に迂回点を置く
            d = np.asarray(b, float) - np.asarray(a, float)
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue
            n = np.array([-d[1], d[0]]) / L
            t = d / L
            for p in (np.asarray(a, float), np.asarray(b, float)):
                for sn in (-1.0, 1.0):
                    for tn in (-1.0, 0.0, 1.0):
                        q = p + sn * n * pad + tn * t * pad
                        if lo <= q[0] <= hi_x and lo <= q[1] <= hi_y:
                            nodes.append(q)
        E = [[] for _ in nodes]
        for i, a in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                pad = pad0 if i == 0 else None      # 始点から出る辺だけ緩める
                if self.free(a, nodes[j], walls, pad):
                    d = dist(a, nodes[j])
                    E[i].append((j, d))
                    E[j].append((i, d))
        D = [math.inf] * len(nodes)
        prev = [-1] * len(nodes)
        D[0] = 0.0
        Q = [(0.0, 0)]
        while Q:
            cost, i = heapq.heappop(Q)
            if cost != D[i]:
                continue
            if i == 1:
                break
            for j, c in E[i]:
                if cost + c < D[j]:
                    D[j] = cost + c
                    prev[j] = i
                    heapq.heappush(Q, (D[j], j))
        if not math.isfinite(D[1]):
            return []
        raw, i = [], 1
        while i >= 0:
            raw.append(nodes[i])
            i = prev[i]
        raw.reverse()
        sm, i = [raw[0]], 0                            # 見通せる最遠点まで間引く
        while i < len(raw) - 1:
            j = len(raw) - 1
            while j > i + 1 and not self.free(raw[i], raw[j], walls):
                j -= 1
            sm.append(raw[j])
            i = j
        return sm

    def target(self, now, pose, goal, walls):
        if not self.path or now - self.plan_t > REPLAN_DT:
            self.path = self.plan(pose, goal, walls)
            self.idx = 0
            self.plan_t = now
        if not self.path:
            return None
        while self.idx < len(self.path) - 1 and dist(pose, self.path[self.idx]) < 45:
            self.idx += 1
        pad0 = min(ROBOT_R + MARGIN, max(ROBOT_R + 3.0, self.clearance(pose, walls) + ROBOT_R - 2.0))
        for i in range(len(self.path) - 1, self.idx, -1):
            if self.free(pose, self.path[i], walls, pad0):
                self.idx = i
                break
        return self.path[self.idx]


# ---------------------------------------------------------------- 走行制御
class Driver:
    """(目標点, 自機の姿勢) → 実機の離散指令 (k, s) に落とす。

    実機APIは 前進/後進/左旋回/右旋回 の4種しかないので
    「向きを合わせる → 進む」の交互駆動にする。

    実機で困る2点への対策
      1. 速度を下げて送ると摩擦に負けて動かないことがある
         → 走るときは常に FIXED_SPEED（既定60）。0 は停止のときだけ
      2. マーカの貼り付け角や駆動の癖で「前進」が思った向きに行かない
         → 実際に動いた向きとマーカの向きのズレを走りながら学習して補正する
    """

    def __init__(self, speed: int, turn_speed: int = 0, learn: bool = True):
        self.turn_speed_raw = turn_speed          # 0 = 前進と同じ
        self.wait_t = 0.0         # 前方の他機を待っている時間[s]
        self.detour_until = 0.0   # 迂回中の期限
        self.detour_key = KEY_LEFT
        self.blocker = None       # 前をふさいでいる相手のID
        self.cal_phase = "pulse"  # 向き合わせ: pulse → settle → 完了
        self.cal_until = 0.0
        self.cal_tries = 0
        self.cal_dir = KEY_FWD
        self.v_est = 150.0        # 実測から学ぶ前進速度[mm/s]
        self.w_est = 800.0        # 実測から学ぶ旋回角速度[deg/s]
        self.last_sent = None     # 直前に送っていた指令
        self.sent_since = 0.0     # その指令が続いている開始時刻
        self.tp_end = 0.0         # 旋回パルスの終了時刻
        self.tp_rest = 0.0        # パルス後の休止の終了時刻
        self.speed = speed
        self.learn = learn
        self.off = 0.0            # 実際に進む向き − マーカの向き [deg]
        self.off_n = 0            # 学習に使ったサンプル数
        self.turning = False      # 旋回フェーズか（ヒステリシス用）
        self.prev = None          # (時刻, 位置, 向き)
        self.no_move = 0.0        # 前進指令なのに動いていない時間[s]
        self.stuck_t = 0.0
        self.best_d = math.inf
        self.back_until = 0.0
        self.rec_until = 0.0
        self.rec_phase = ""
        self.rec_turn = KEY_RIGHT
        self.rec_n = 0

    @property
    def turn(self) -> int:
        """旋回に使う速度。--turn-speed 0 なら前進と同じ。"""
        return self.turn_speed_raw if self.turn_speed_raw > 0 else self.speed

    def bump(self, d_fwd: int = 0, d_turn: int = 0) -> None:
        """走行中に速度を微調整する（キー操作から呼ぶ）。"""
        if d_fwd:
            self.speed = int(max(5, min(100, self.speed + d_fwd)))
        if d_turn:
            base = self.turn
            self.turn_speed_raw = int(max(5, min(100, base + d_turn)))

    # ---- 観測: 実際にどう動いたかを見て向きのズレを学習する ----
    def observe(self, now, pose, head, sent):
        if sent != self.last_sent:
            self.last_sent = sent
            self.sent_since = now          # 指令が変わった時刻を覚える
        if self.prev is None:
            self.prev = (now, np.array(pose, float), head)
            return
        pt, pp, ph = self.prev
        dt2 = now - pt
        if dt2 < 0.12:
            return
        d = np.array(pose, float) - pp
        mv = float(np.linalg.norm(d))
        k, sp = sent
        # 「同じ指令がその区間ずっと続いていた」ときだけ学習に使う。
        # 旋回パルスと休止をまたいだ区間や、後退・退避中の観測を混ぜると
        # マーカが後方に付いている分だけ弧を描くので、向きのズレが暴れる。
        clean = (now - self.sent_since) >= dt2 + 0.02 and dt2 <= 0.40

        if sp > 0 and k == KEY_FWD:
            if mv >= 15.0 and clean:
                self.v_est = self.v_est * 0.8 + (mv / dt2) * 0.2
                mdir = math.degrees(math.atan2(d[1], d[0]))
                diff = norm180(mdir - ph)
                if self.learn:
                    g = 1.0 if self.off_n == 0 else (0.5 if self.off_n < 5 else LEARN_GAIN)
                    self.off = norm180(self.off + g * norm180(diff - self.off))
                    self.off_n += 1
                self.no_move = 0.0
            elif mv >= MOVE_EPS:
                self.no_move = 0.0
            else:
                self.no_move += dt2
        elif sp > 0 and k in (KEY_LEFT, KEY_RIGHT):
            dh = abs(norm180(head - ph))
            if dh > 2.0 and clean:
                self.w_est = self.w_est * 0.8 + (dh / dt2) * 0.2
            self.no_move = 0.0
        else:
            self.no_move = 0.0
        self.prev = (now, np.array(pose, float), head)

    def heading(self, head):
        """マーカの向きを、実際に進む向きに補正した値。"""
        return norm180(head + self.off)

    def escape(self, now, dt, pose, head, planner, walls):
        """経路が引けない（壁に寄りすぎ・接触）ときに抜け出す。

        壁の真上に乗ると「離れる向き」が計算できず前後に振動するので、
        幾何に頼らず「後退 → 旋回」の時間制シーケンスで確実に抜ける。
        """
        if now >= self.rec_until:
            self.rec_n += 1
            self.rec_phase = "back"
            self.rec_until = now + min(0.6 + 0.3 * self.rec_n, 1.8)
            self.rec_turn = KEY_RIGHT if (self.rec_n % 2) else KEY_LEFT
            return KEY_BACK, self.speed, f"退避{self.rec_n}: 後退"
        if self.rec_phase == "back":
            return KEY_BACK, self.speed, f"退避{self.rec_n}: 後退"
        return self.rec_turn, self.speed, f"退避{self.rec_n}: 旋回"

    def escape_tick(self, now):
        if self.rec_phase == "back" and now >= self.rec_until:
            self.rec_phase = "turn"
            self.rec_until = now + 0.45
        elif self.rec_phase == "turn" and now >= self.rec_until:
            self.rec_phase = ""
            self.rec_until = 0.0

    def calib(self, now, pose, clearance):
        if not self.learn:
            return None          # 学習しないなら向き合わせも不要
        """開始時の向き合わせ。短い前進パルス→停止→計測、を数回だけ行う。

        いきなり本走行すると、マーカを大きくずらして貼っていた場合に
        学習し終わる前に壁や場外へ行ってしまう。
        45mm 程度ずつ動かして測るので、どの向きでも安全。
        戻り値: (k, s, note) / 完了したら None
        """
        if self.off_n >= 2 or self.cal_tries >= CAL_TRIES:
            return None
        if now >= self.cal_until:
            if self.cal_phase == "pulse":
                self.cal_phase = "settle"
                self.cal_until = now + CAL_SETTLE
            else:
                self.cal_phase = "pulse"
                self.cal_tries += 1
                self.cal_until = now + CAL_PULSE
                # 壁や縁に寄ってきたら、次のパルスは逆向きにする
                self.cal_dir = KEY_BACK if clearance < 25 else KEY_FWD
        if self.cal_phase == "settle":
            return KEY_FWD, 0, f"向き合わせ: 計測 ({self.cal_tries}/{CAL_TRIES})"
        return self.cal_dir, self.speed, f"向き合わせ: 微動 ({self.cal_tries}/{CAL_TRIES})"

    def turn_step(self, now, err):
        """旋回を短いパルスで出す。

        実機は旋回が速い（実測 0.05 秒で約 40 度）ので、連続で回すと
        1制御周期のあいだに回り過ぎる。必要な角度ぶんだけ回して、
        止めて測り直す。パルス時間は
            必要角度 ÷ (TURN_DEG_PER_PULSE / TURN_PULSE)
        で決める。回り過ぎるなら TURN_PULSE を小さく、
        回らないなら TURN_DEG_PER_PULSE を小さくする。
        """
        ep = getattr(self, "err_p", err)
        k = KEY_RIGHT if ep > 0 else KEY_LEFT
        if TURN_PULSE >= 1.0:
            return k, self.turn, f"旋回(連続) {ep:+.0f}deg"
        if now < self.tp_end:
            return k, self.turn, f"旋回 {ep:+.0f}deg"
        if now < self.tp_rest:
            return KEY_FWD, 0, f"旋回の合間に測定 {ep:+.0f}deg"
        rate = max(30.0, TURN_DEG_PER_PULSE / max(0.01, TURN_PULSE))   # deg/s
        dur = min(TURN_PULSE * 2.0, max(0.02, abs(ep) / rate))
        self.tp_end = now + dur
        self.tp_rest = self.tp_end + TURN_GAP
        return k, self.turn, f"旋回 {ep:+.0f}deg ({dur*1000:.0f}ms)"

    def front_blocker(self, pose, he, others):
        """進行方向の前方近くにいる他機を返す（居なければ None）。"""
        best, bd = None, 1e9
        for i, p in others:
            d = dist(pose, p)
            if d > AVOID_R:
                continue
            ang = norm180(math.degrees(math.atan2(p[1] - pose[1], p[0] - pose[0])) - he)
            if abs(ang) <= AVOID_FOV and d < bd:
                bd, best = d, (i, d, ang)
        return best

    def step(self, now, dt, pose, head, target, dgoal, clearance=999.0, others=()):
        self.rec_n = 0
        cal = self.calib(now, pose, clearance)        # まず向きのズレを測る
        if cal is not None:
            return cal
        if now < self.back_until:
            return KEY_BACK, self.speed, "後退中"

        # 進んでいなければ後退して仕切り直す
        if dgoal > self.best_d - 1.0:
            self.stuck_t += dt
        else:
            self.stuck_t = 0.0
        self.best_d = min(self.best_d, dgoal)
        if self.stuck_t > 2.5:
            self.stuck_t = 0.0
            self.best_d = dgoal
            self.back_until = now + 0.7
            return KEY_BACK, self.speed, "詰まり→後退"

        # 前進指令なのに動いていないときは、素直に後退して向きを変える
        if self.no_move > 1.2:
            self.no_move = 0.0
            self.back_until = now + 0.5
            return KEY_BACK, self.speed, "前進しても動かない→後退"

        he = self.heading(head)                       # 学習で補正した向き
        err = norm180(math.degrees(math.atan2(target[1] - pose[1], target[0] - pose[0])) - he)

        # 遅れぶんを先読みした誤差で判定する。
        # 見えている誤差で止めると、その間に回った分だけ必ず回り過ぎる。
        # 先読みは効かせすぎると「旋回に入らない」ので上限をかける
        lead = min(self.w_est * TURN_SAFETY * TURN_LEAD_S, TURN_LEAD_MAX, abs(err) * 0.5)
        err_p = err - math.copysign(lead, err)

        # ヒステリシス: 旋回に入ったら TURN_OUT まで合わせてから前進する。
        if self.turning:
            if abs(err_p) <= TURN_OUT:
                self.turning = False
                self.tp_end = self.tp_rest = 0.0
        elif abs(err) >= TURN_IN:
            self.turning = True
        self.err_p = err_p

        if self.turning:
            self.wait_t = 0.0
            self.blocker = None
            return self.turn_step(now, err)

        # --- 前方に他機が居るとき ---
        # まず止まって待つ。相手が動くものなら数百msでどく。
        # それでもどかなければ、ぶつけずに横へ迂回する。
        if now < self.detour_until:
            return self.detour_key, self.turn, f"迂回中 (ID{self.blocker})"
        blk = self.front_blocker(pose, he, others)
        if blk is not None:
            bid, bd, bang = blk
            self.blocker = bid
            self.wait_t += dt
            if self.wait_t > AVOID_WAIT:
                self.wait_t = 0.0
                self.detour_until = now + 0.6
                self.detour_key = KEY_LEFT if bang > 0 else KEY_RIGHT   # 相手の反対へ回る
                return self.detour_key, self.turn, f"どかないので迂回 (ID{bid})"
            return KEY_FWD, 0, f"前方にID{bid} {bd:.0f}mm→待機 {self.wait_t:.1f}s"
        self.wait_t = 0.0
        self.blocker = None
        return KEY_FWD, self.speed, f"直進 {err:+.0f}deg c={clearance:.0f}"


# ---------------------------------------------------------------- 描画
WIN = "BUGC2"


def draw_overlay(frame, Hi, info):
    """壁・経路・ゴール・軌跡を映像に重ねる（元解像度の座標で描く）。"""
    if Hi is None:
        return
    for a, b in info["walls"]:
        cv2.line(frame, to_image(Hi, a), to_image(Hi, b), (150, 150, 150), 4)
    path = info["path"]
    for i in range(1, len(path)):
        cv2.line(frame, to_image(Hi, path[i - 1]), to_image(Hi, path[i]), (255, 160, 0), 2)
    for p in path:
        cv2.circle(frame, to_image(Hi, p), 4, (255, 160, 0), -1)
    if info["target"] is not None:
        cv2.circle(frame, to_image(Hi, info["target"]), 9, (0, 200, 255), 2)
    if info["goal"] is not None:
        c = to_image(Hi, info["goal"])
        e = to_image(Hi, info["goal"] + np.array([GOAL_R, 0.0]))
        cv2.circle(frame, c, int(max(6, dist(c, e))), (0, 220, 120), 2)
    cp = info.get("cpose")
    if cp is not None:
        q = to_image(Hi, cp)
        cv2.drawMarker(frame, q, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    for i, p in info.get("others", ()):
        q = to_image(Hi, p)
        cv2.circle(frame, q, 26, (120, 120, 255), 2)
        cv2.putText(frame, str(i), (q[0] + 20, q[1] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (120, 120, 255), 2, cv2.LINE_AA)
    tr = info["trail"]
    for i in range(1, len(tr)):
        cv2.line(frame, to_image(Hi, tr[i - 1]), to_image(Hi, tr[i]), (0, 200, 255), 2)
    # 机の外周（キャリブが効いているかの目視確認用）
    box = [np.array([0.0, 0.0]), np.array([FIELD_W, 0.0]),
           np.array([FIELD_W, FIELD_H]), np.array([0.0, FIELD_H])]
    for i in range(4):
        cv2.line(frame, to_image(Hi, box[i]), to_image(Hi, box[(i + 1) % 4]), (255, 120, 0), 1)


def fit_view(frame, max_w: int, max_h: int):
    """画面からはみ出さないように表示用だけ縮小する（検出は元解像度のまま）。"""
    h, w = frame.shape[:2]
    if max_w <= 0 or max_h <= 0:
        return frame
    k = min(max_w / w, max_h / h, 1.0)
    if k >= 0.999:
        return frame
    return cv2.resize(frame, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA)


def draw_text(view, lines):
    """文字は縮小後に描く（縮小しても読める大きさを保つ）。

    帯は半透明にする。不透明だと左上の四隅マーカ(20)が隠れて、
    キャリブレーションできているのに見えなくなってしまう。
    """
    n = len(lines)
    h = 10 + 21 * n
    band = view[0:h, :]
    cv2.addWeighted(band, 0.30, np.zeros_like(band), 0.70, 0, dst=band)
    for i, (txt, col) in enumerate(lines):
        y = 22 + 21 * i
        cv2.putText(view, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- メイン
def main() -> None:
    global TURN_PULSE, TURN_GAP, TURN_LEAD_S, TURN_DEG_PER_PULSE      # 実行時オプションで上書きする
    ap = argparse.ArgumentParser(description="BUGC2 本番コントローラ")
    ap.add_argument("--camera", default="http://192.168.100.24/video_feed",
                    help="MJPEG URL または USBカメラ番号（PC内蔵カメラなら 0）")
    ap.add_argument("--robot", default="http://192.168.100.250", help="BUGC2 のベースURL")
    ap.add_argument("--id", type=int, default=2, dest="my_id", help="自機のマーカID")
    ap.add_argument("--speed", type=int, default=FIXED_SPEED,
                    help=f"走行速度 s (5-100)。既定 {FIXED_SPEED} 固定")
    ap.add_argument("--turn-speed", type=int, default=TURN_SPEED, dest="turn_speed",
                    help=f"旋回速度 s（既定 {TURN_SPEED}）。0 なら走行と同じ")
    ap.add_argument("--turn-pulse", type=float, default=TURN_PULSE, dest="turn_pulse",
                    help=f"旋回1パルスの長さ[s]（既定 {TURN_PULSE}）。1以上で連続旋回")
    ap.add_argument("--turn-deg", type=float, default=TURN_DEG_PER_PULSE, dest="turn_deg",
                    help=f"1パルスで回る角度[deg]（既定 {TURN_DEG_PER_PULSE:.0f}）。実機を見て直す")
    ap.add_argument("--turn-gap", type=float, default=TURN_GAP, dest="turn_gap",
                    help=f"パルス後に止まって測る時間[s]（既定 {TURN_GAP}）")
    ap.add_argument("--turn-lead", type=float, default=TURN_LEAD_S, dest="turn_lead",
                    help=f"遅れの先読み時間[s]（既定 {TURN_LEAD_S}）。回り過ぎるなら大きく")
    ap.add_argument("--auto", action="store_true", help="起動直後から自動走行")
    ap.add_argument("--headless", action="store_true", help="映像ウィンドウを出さない")
    ap.add_argument("--duration", type=float, default=0.0, help="この秒数で自動終了（0=無制限）")
    ap.add_argument("--dry-run", action="store_true", help="実機へ送らず表示だけ")
    ap.add_argument("--verbose", action="store_true", help="1秒ごとに状態をコンソールへ出す")
    ap.add_argument("--check", action="store_true", help="機体に届くかだけ確認して終了")
    ap.add_argument("--scan", nargs="?", const="", default=None, metavar="192.168.100",
                    help="ネットワークを総当たりして機体を探して終了")
    ap.add_argument("--no-robot", action="store_true", help="機体には一切送らず、映像と経路だけ確認")
    ap.add_argument("--view", default="1280x720", metavar="WxH",
                    help="表示ウィンドウの最大サイズ。'0' で縮小しない（既定 1280x720）")
    ap.add_argument("--marker-back", type=float, default=MARKER_BACK, dest="marker_back",
                    help=f"マーカ中心から機体の回転中心までの前向き距離[mm]（既定 {MARKER_BACK:.0f}）"
                         "。マーカが後方に付いているほど大きく")
    ap.add_argument("--blind", type=float, default=BLIND_S,
                    help=f"自機を見失っても指令から推測して走り続ける時間[s]（既定 {BLIND_S}、0で即停止）")
    ap.add_argument("--learn", action="store_true",
                    help="「マーカの向き」と「実際に進む向き」のズレを走りながら学習する（既定オフ）")
    ap.add_argument("--head-offset", type=float, default=0.0, dest="head_offset",
                    help="向きの補正[deg]を手で与える。マーカを斜めに貼ったときだけ使う")
    ap.add_argument("--drop", type=float, default=0.0, metavar="P",
                    help="動作確認用: 自機マーカの検出を確率Pでわざと落とす（0〜1）")
    ap.add_argument("--shot", default="", metavar="PATH",
                    help="表示中の画面を1枚PNGで保存して終了（見え方の確認・共有用）")
    a = ap.parse_args()
    print(f"test.py {VERSION}")
    print(f"このPCのIP: {local_ip()}")

    if a.scan is not None:
        hosts = scan_robots(a.scan or None)
        if hosts:
            print("応答あり:")
            for h in hosts:
                print(f"  http://{h}   ←  --robot http://{h}")
        else:
            print("見つかりません。PCと機体が同じWi-Fiに居るか確認してください。")
        return

    if a.check:
        ok, msg = check_robot(a.robot)
        print(("機体 " + msg) if ok else ("機体 " + msg))
        if not ok:
            print("\n  確認すること:")
            print("   1) PC の Wi-Fi が機体と同じネットワークか（CQ-WS / CQ-WS-2-4G）")
            print("   2) 機体の電源が入っているか")
            print("   3) IP が変わっていないか →  python test.py --scan")
        return

    if a.no_robot:
        a.dry_run = True

    source = int(a.camera) if a.camera.isdigit() else a.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"カメラを開けません: {a.camera}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    robot = Robot(a.robot, dry_run=a.dry_run)
    TURN_PULSE, TURN_GAP, TURN_LEAD_S = a.turn_pulse, a.turn_gap, a.turn_lead
    TURN_DEG_PER_PULSE = a.turn_deg
    planner, driver = Planner(), Driver(a.speed, a.turn_speed, learn=a.learn)
    driver.off = a.head_offset
    print(f"マーカ→回転中心 前方 {a.marker_back:.0f}mm  向き補正 {a.head_offset:+.0f}deg"
          + ("  学習ON" if a.learn else ""))
    print(f"走行速度={a.speed}  旋回速度={driver.turn}  "
          f"旋回パルス={TURN_PULSE*1000:.0f}ms/{TURN_DEG_PER_PULSE:.0f}deg  "
          f"休止={TURN_GAP*1000:.0f}ms  先読み={TURN_LEAD_S*1000:.0f}ms")
    H = Hi = None
    mode = "AUTO" if a.auto else "STOP"
    lost = 0
    trail: list[np.ndarray] = []
    static: dict[int, np.ndarray] = {}      # 動かないマーカ（ゴール・壁）の記憶
    t0 = time.monotonic()
    started = None
    finished = False
    fps_t, fps_n, fps = time.monotonic(), 0, 0.0
    last_t = vb_t = time.monotonic()
    manual_until = 0.0
    manual_key = ""
    last_cmd = None
    tracker = Tracker()
    st_cand: dict[int, tuple] = {}   # 静止物が「置き直された」かの判定用
    self_pose = None                 # 見失っている間も保持する自機位置
    self_head = 0.0
    blind = 0.0                      # 見失ってからの経過[s]
    rejected = 0                     # あり得ない飛びとして棄却した回数
    fail_n = 0                       # フレーム取得の連続失敗回数
    first_show = True
    shown_size = False

    robot_ok = True
    if not a.dry_run:
        robot_ok, msg = check_robot(a.robot, timeout=2.0)
        print("機体 " + msg)
        if not robot_ok:
            print("  → 機体には届きません。映像と経路の確認は続けられます。")
            print("     IP が分からないときは:  python test.py --scan")
    if a.view.strip() in ("0", ""):
        view_w = view_h = 0
    else:
        try:
            view_w, view_h = (int(v) for v in a.view.lower().split("x"))
        except ValueError:
            view_w, view_h = 1280, 720
    if not a.headless:
        try:
            # WINDOW_NORMAL にしておくと、ウィンドウを手で自由にリサイズできる
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(WIN, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        except cv2.error as e:
            print("ウィンドウを開けません（GUI無しの OpenCV かもしれません）:", e)
            print("  pip install opencv-contrib-python  を入れ直すか、--headless で実行してください")
            a.headless = True
    print(f"実機: {a.robot}/cmd?k=..&s=..   {'(dry-run)' if a.dry_run else ''}")
    print("Enter=完全自律スタート  Space=停止  WASD=手動  R=再キャリブ  P=キー確認  Q=終了")
    print("速度調整: +/- で走行±5、[ ] で旋回±5、数字1〜9でその値×10（6→60）、0で旋回を走行と同じに")
    print("ウィンドウは手でリサイズできます。全画面で見たいときは --view 0")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # ストリームは普通に切れる。終了せずに開き直して自律を続ける
                robot.stop()
                fail_n += 1
                if fail_n == 1 or fail_n % 20 == 0:
                    print(f"フレームを取得できません（{fail_n}回目）→ 開き直します: {a.camera}")
                cap.release()
                time.sleep(min(0.3 * fail_n, 2.0))
                cap = cv2.VideoCapture(source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            fail_n = 0
            if not shown_size:
                shown_size = True
                h0, w0 = frame.shape[:2]
                print(f"映像サイズ: {w0}x{h0}"
                      + (f" → 表示は最大 {view_w}x{view_h} に縮小（検出は {w0}x{h0} のまま）"
                         if view_w and (w0 > view_w or h0 > view_h) else ""))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marks = detect(gray)
            if a.drop > 0 and a.my_id in marks and random.random() < a.drop:
                marks.pop(a.my_id)          # わざと見失わせて惰行の挙動を確認する

            if H is None:
                H, Hi = calibrate(marks)
                if H is not None:
                    print("四隅マーカ 20〜23 でキャリブレーションしました")

            now = time.monotonic()
            dt = min(0.2, max(0.001, now - last_t))
            last_t = now

            mm, degs = {}, {}
            if H is not None:
                for i, q in marks.items():
                    mm[i] = to_field(H, q.mean(axis=0))
                    degs[i] = pose_of(H, q)[1]

            # --- 追跡層に流し込む。あり得ない飛びはここで棄却される ---
            for i, p in mm.items():
                if not tracker.update(now, i, p, degs.get(i, 0.0)):
                    rejected += 1

            # --- ゴールと壁は動かないので覚えておく ---
            # 自機がゴールマーカに乗ると隠れて到達判定ができなくなる。
            # 置き直された場合だけ、3回続けて同じ場所に見えたら更新する。
            for i, p in mm.items():
                if i != 0 and not (30 <= i <= 96):
                    continue
                old_p = static.get(i)
                if old_p is None:
                    static[i] = p
                elif dist(old_p, p) <= STATIC_MOVE:
                    static[i] = old_p * 0.85 + p * 0.15          # 平滑化
                    st_cand.pop(i, None)
                else:
                    c = st_cand.get(i)
                    if c is not None and dist(c[0], p) <= STATIC_MOVE:
                        st_cand[i] = (p, c[1] + 1)
                    else:
                        st_cand[i] = (p, 1)
                    if st_cand[i][1] >= 3:
                        static[i] = p
                        st_cand.pop(i, None)
                        print(f"ID{i} が動いたので置き直しました")

            known = dict(static)
            known.update(mm)
            goal = known.get(0)
            walls = walls_from(known)
            # 他機は見失っても HOLD_S 秒は居るものとして扱う（点滅で突っ込まないため）
            others = [(i, p) for i, p in
                      tracker.alive(now, list(ROBOT_IDS) + list(JAM_IDS), HOLD_S)
                      if i != a.my_id]

            # --- 自機の姿勢: 見えていれば観測、見えなければ指令から推測 ---
            cpose = None                     # 制御に使う点（回転中心）
            seen = tracker.get(now, a.my_id, hold=0.0)
            if seen is not None and a.my_id in mm:
                pose, head = seen[0], seen[1]
                driver.observe(now, pose, head, robot.cur)
                blind = 0.0
                self_pose, self_head = np.array(pose, float), head
                cp0 = control_point(pose, driver.heading(head), a.marker_back)
                if not trail or dist(trail[-1], cp0) > 5:
                    trail.append(cp0)
                    trail[:] = trail[-600:]
                lost = 0
            elif self_pose is not None:
                blind += dt
                lost += 1
                if blind <= a.blind:
                    # 一時的な見失い。出している指令から進んだぶんを足して走り続ける
                    self_pose, self_head = driver.dead_reckon(self_pose, self_head,
                                                              robot.cur, dt)
                    pose, head = self_pose, self_head
                else:
                    pose = head = None            # 長く見えない → 止まって待つ
            else:
                lost += 1
                pose = head = None

            if pose is not None:
                cpose = control_point(pose, driver.heading(head), a.marker_back)

            target, note = None, ""

            if mode == "AUTO":
                if pose is None:
                    # 長く見えない間は止めて待つ。見えたら自動で走行を再開する
                    robot.stop()
                    last_cmd = None
                    planner.path = []
                    note = (f"自機ID{a.my_id}が見えない→停止して復帰待ち "
                            f"({blind:.1f}s)")
                elif goal is None:
                    robot.stop()
                    note = "ゴール(ID0)を一度も見ていない"
                else:
                    dgoal = dist(cpose, goal)
                    if dgoal < GOAL_R:
                        robot.stop()
                        last_cmd = None
                        finished = True
                        mode = "STOP"
                        el = now - (started or now)
                        note = f"ゴール! {el:.2f}s"
                        print(f"*** ゴール  {el:.2f} 秒")
                    else:
                        clear = planner.clearance(cpose, walls)
                        driver.escape_tick(now)
                        target = None if driver.rec_phase else planner.target(now, cpose, goal, walls)
                        if target is None:
                            # 経路が引けない／退避中は、後退→旋回で確実に抜ける
                            k, s_, note = driver.escape(now, dt, cpose, head, planner, walls)
                            robot.set(k, s_)
                            last_cmd = (k, s_)
                            planner.path = []
                        else:
                            k, s, note = driver.step(now, dt, cpose, head, target, dgoal,
                                                     clear, others)
                            robot.set(k, s)
                            last_cmd = (k, s)
                            if started is None:
                                started = now
            elif mode != "MANUAL":
                robot.stop()

            fps_n += 1
            if now - fps_t >= 0.5:
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0

            # ---- 表示 ----
            col_mode = {"AUTO": (0, 255, 120), "MANUAL": (0, 200, 255)}.get(mode, (200, 200, 200))
            ck, cs2 = robot.cur
            moving = cs2 != 0
            lines = [
                (f"[{mode}] {note}", col_mode),
                (f"送信中: k={ck} s={cs2}"
                 + ("  ← 動作中" if moving else "  ← 停止")
                 + (f"   （有効期限切れで自動停止）" if robot.expired else "")
                 + f"   速度={driver.speed}/旋回{driver.turn}"
                 + (f"   向き補正={driver.off:+.0f}deg({driver.off_n})" if driver.off_n else "   向き補正=学習中"),
                 (0, 80, 255) if moving else (120, 255, 120)),
                (f"cal={'OK' if H is not None else '--'}  ids={sorted(marks)}"
                 + (f"  記憶={sorted(k for k in static if k not in mm)}"
                    if any(k not in mm for k in static) else ""), (255, 255, 255)),
            ]
            # 自機とゴールは別々に出す（片方が欠けたときにもう片方まで -- になると誤解を招く）
            if pose is not None:
                g_txt = (f"goal_d={dist(cpose if cpose is not None else pose, goal):.0f}mm"
                         if goal is not None
                         else "goal=ID0未確認")
                _cp = cpose if cpose is not None else pose
                lines.append((f"自機ID{a.my_id} 中心=({_cp[0]:.0f},{_cp[1]:.0f}) head={head:+.0f}deg "
                              + ("[推測]" if blind > 0 else "") +
                              f" {g_txt}  fps={fps:.0f}",
                              (255, 255, 255) if goal is not None else (0, 200, 255)))
            else:
                lines.append((f"自機ID{a.my_id}=見えない (lost {lost})  "
                              f"{'goal=OK' if goal is not None else 'goal=ID0未確認'}  "
                              f"fps={fps:.0f}", (0, 165, 255)))
            if others:
                lines.append(("他機: " + "  ".join(
                    f"ID{i}({'相手' if i in ROBOT_IDS else '妨害'})"
                    + (f"{dist(cpose, p):.0f}mm" if cpose is not None else "")
                    for i, p in others), (200, 200, 255)))
            lines.append((f"tx={robot.sent} err={robot.errors}  棄却{rejected}  "
                          f"実測 v={driver.v_est:.0f}mm/s w={driver.w_est:.0f}deg/s", (180, 180, 180)))
            if robot.errors and robot.sent == 0:
                lines.append(("機体に届いていません: " + robot.last_err[:55], (0, 0, 255)))
                lines.append(("同じWi-Fiか / 電源 / IP を確認  (python test.py --scan)", (0, 0, 255)))
            elif robot.errors and robot.last_err:
                lines.append((f"直近の送信エラー({robot.errors}件): " + robot.last_err[:50], (0, 140, 255)))

            if (not a.headless) or a.shot:
                if marks:
                    ids = np.array([[i] for i in marks], dtype=np.int32)
                    cs = [marks[i].reshape(1, 4, 2) for i in marks]
                    cv2.aruco.drawDetectedMarkers(frame, cs, ids)
                draw_overlay(frame, Hi, {"walls": walls, "path": planner.path,
                                         "target": target, "goal": goal, "trail": trail,
                                         "others": others, "cpose": cpose})
                view = fit_view(frame, view_w, view_h)          # 見切れないように縮小
                draw_text(view, lines)
                bk, bs = robot.cur
                tag = f"{bk.upper()} {bs}" if bs else "STOP"
                tcol = (0, 60, 255) if bs else (110, 240, 110)
                cv2.putText(view, tag, (view.shape[1] - 150, view.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(view, tag, (view.shape[1] - 150, view.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, tcol, 2, cv2.LINE_AA)
                if a.shot:
                    cv2.imwrite(a.shot, view)
                    print(f"画面を保存しました: {a.shot}  ({view.shape[1]}x{view.shape[0]})")
                    break
                cv2.imshow(WIN, view)
                if first_show:
                    first_show = False
                    try:
                        cv2.resizeWindow(WIN, view.shape[1], view.shape[0])
                    except cv2.error:
                        pass
                key = cv2.waitKey(1) & 0xFF
            if a.headless and not a.shot:
                key = 255
                time.sleep(0.02)

            # ---- キー ----
            if key in (ord("q"), 27):
                break
            elif key in (13, 10):                       # Enter
                mode, finished = "AUTO", False
                last_cmd, blind = None, 0.0
                driver.cal_phase, driver.cal_until, driver.cal_tries = "pulse", 0.0, 0
                driver.off_n = 0
                planner.path, planner.plan_t = [], -99.0
                driver.best_d, driver.stuck_t, driver.back_until = math.inf, 0.0, 0.0
                started = None
                print("自動走行 開始")
            elif key == ord(" "):
                mode = "STOP"
                last_cmd = None
                robot.stop()
                print("停止")
            elif key in (ord("w"), ord("a"), ord("s"), ord("d")):
                mode = "MANUAL"
                manual_key = {ord("w"): KEY_FWD, ord("a"): KEY_LEFT,
                              ord("s"): KEY_BACK, ord("d"): KEY_RIGHT}[key]
                # キーリピートは歯抜けで届くので、少し保持してから止める
                manual_until = now + 0.30
            elif key in (ord("+"), ord("=")):
                driver.bump(d_fwd=+5)
                print(f"速度 {driver.speed} / 旋回 {driver.turn}")
            elif key in (ord("-"), ord("_")):
                driver.bump(d_fwd=-5)
                print(f"速度 {driver.speed} / 旋回 {driver.turn}")
            elif key == ord("]"):
                driver.bump(d_turn=+5)
                print(f"速度 {driver.speed} / 旋回 {driver.turn}")
            elif key == ord("["):
                driver.bump(d_turn=-5)
                print(f"速度 {driver.speed} / 旋回 {driver.turn}")
            elif ord("1") <= key <= ord("9"):
                driver.speed = (key - ord("0")) * 10          # 6 を押せば 60
                print(f"速度 {driver.speed} / 旋回 {driver.turn}")
            elif key == ord("0"):
                driver.turn_speed_raw = 0                     # 旋回速度を前進と同じに戻す
                print(f"速度 {driver.speed} / 旋回 {driver.turn}（前進と同じ）")
            elif key == ord("r"):
                H = Hi = None
                static.clear()
                print("キャリブレーションと記憶をやり直します")
            elif key == ord("p"):
                print("キー割り当ての確認:")
                robot.probe()
            if mode == "MANUAL":
                if now <= manual_until and manual_key:
                    robot.set(manual_key, driver.speed)   # 毎回出し直す（TTL対策）
                else:
                    manual_key = ""
                    robot.stop()

            if a.verbose and now - vb_t >= 1.0:
                vb_t = now
                print(" | ".join(t for t, _ in lines[:3]))

            if a.duration and now - t0 > a.duration:
                break
    finally:
        robot.close()
        cap.release()
        if not a.headless:
            cv2.destroyAllWindows()
        print(f"終了  送信 {robot.sent} 回 / 失敗 {robot.errors} 回"
              + (f" / 最後のエラー: {robot.last_err}" if robot.last_err else ""))


if __name__ == "__main__":
    main()

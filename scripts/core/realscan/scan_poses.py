#!/usr/bin/env python3
"""보드 추적 스캔 자세 — 다녀 본 시점 방향을 물체 쪽으로 당겨 pose_tour --poses YAML 로 쓴다 (PiPER).

시점 방향은 이전 촬영 run(capture.json 관절값)이나 자세 YAML 에서 가져온다 — 팔이 실제로 다녀 본 방향이고,
방향은 로봇(base) 기준으로 유지한다. 시점마다 카메라(depth)를 물체 중심에서 --distances 순서(가까운 것부터)로
놓고, 안 되면 높이각을 --raise-deg 만큼 바꿔 IK 를 푼다(시드 = 그 시점의 관절값). 남기는 조건:
  관절 한계(여유) · 손목 뻗음 ≤ --max-reach · 시드 팔 모양에서 --max-move-deg 이내(손목 뒤집힘 방지) ·
  손가락이 물체·받침대 상자와 보드에서 --finger-clear 이상 · 보이는 보드 코너 ≥ --min-corners
  (물체·받침대 상자와 손가락이 가린 코너는 뺀다).
벌린 그리퍼 손가락은 카메라 앞 9–13 cm·화면 아래 양쪽에 찍힌다 — 낮은 시점에서 보드에 닿는 것이 가까이 찍는
것을 막는 주원인이었다. 손가락 모양은 --fingers-from run 의 depth 에서 카메라 앞 13 cm 안 점(카메라 좌표)과
그 화면 영역으로 둔다. 실제 검출 코너는 이 계산의 ~55–60% 였다(2026-10-20) — 기본 26 ≈ 실제 15.
손가락 여유 2.5 cm 는 PiPER 팔 FK 오차(자세에 따라 ~14 mm)를 넘게 잡은 값이다.

작업 좌표(보드 중심, x 긴 변, z 위)의 base 위치는 --work-frame: 앱 run 폴더(capture.json 의 base_T_world) 또는
board_check.py --scan-board --save-frame 으로 잰 JSON. 보드를 돌렸으면 다시 재서 넣는다 — 시점 방향이 base
기준이라 같은 자세를 물체가 옮겨진 만큼 평행이동한 결과가 된다. 물체·받침대 상자는 작업 좌표로 준다.
끝에 pose_tour 에 줄 충돌 상자(--obstacle, base 기준)를 같이 출력한다.

회전판: --view 높이각 방위 로 시점 방향을 직접 주고(방위 0 = 물체에서 로봇 쪽, base 기준) --turns 4 를 주면,
판(보드·받침대·물체)을 물체 중심을 지나는 수직축으로 90° 씩 돌린 4 경우 모두에서 코너·손가락을 검사한다 —
물체를 회전판 중심에 두면 같은 자세를 판을 돌릴 때마다 그대로 쓴다. 시드(--views-from/--poses, 여럿 가능)는
IK 출발점으로만 쓰고, 방향이 가까운 것부터 고른다.

    python scripts/core/realscan/scan_poses.py --views-from data/blue_cube/scan/20261020_020042 \\
        --work-frame output/piper_scan_poses/frame.json --object 0 0 0.068 0.056 0.056 0.058 \\
        --keep-out 0 0 0.021 0.05 0.05 0.042 --fingers-from data/blue_cube/scan/20261020_020042 \\
        --out output/piper_scan_poses/stand.yaml
    # 회전판: 위 1 + 45° 2 + 20° 2, 판 90° 씩 4번
    python scripts/core/realscan/scan_poses.py --poses output/piper_scan_poses/*.yaml \\
        --view 90 0 --view 45 -60 --view 45 -105 --view 20 -30 --view 20 -75 --turns 4 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from core.realscan.calib_poses import FastFK, LIMIT_MARGIN_DEG, solve_ik, wrist_reach  # noqa: E402
from core.realscan.capture import transform  # noqa: E402
from core.realscan.piper import PiperFK  # noqa: E402

FINGER_RANGE_M = (0.03, 0.13)      # 카메라 앞 이 거리 안 depth 점 = 손가락 (물체는 그보다 멀다)
FINGER_MASK_DILATE_PX = 25


def box_distance(points, centre, size):
    """점 → 상자 표면까지 거리 (안쪽은 음수). size 는 전체 크기."""
    q = np.abs(np.asarray(points) - centre) - np.asarray(size) / 2
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def segment_hits_box(a, b, centre, size):
    """선분 a→b 가 상자를 지나는가 (slab)."""
    lo, hi = np.asarray(centre) - np.asarray(size) / 2, np.asarray(centre) + np.asarray(size) / 2
    d = b - a
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if not lo[k] <= a[k] <= hi[k]:
                return False
            continue
        u, v = (lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]
        t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
    return t0 < t1


def board_corners(spec):
    """보드 안쪽 코너 — 작업 좌표(보드 중심 원점, z=0)."""
    sx, sy, sq = spec["squares_x"], spec["squares_y"], spec["square_m"]
    gx, gy = np.meshgrid((np.arange(1, sx) - sx / 2) * sq, (np.arange(1, sy) - sy / 2) * sq)
    return np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]


def spin(angle, centre):
    """작업 좌표에서 centre 를 지나는 z 축으로 angle[rad] 돌리는 변환 (회전판)."""
    c, s_ = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:2, :2] = [[c, -s_], [s_, c]]
    T[:2, 3] = np.asarray(centre)[:2] - T[:2, :2] @ np.asarray(centre)[:2]
    return T


def view_directions(views, base_T_world, centre):
    """(높이각, 방위)[deg] → 작업 좌표 시점 방향(물체→카메라). 방위 0 = 물체에서 로봇 base 쪽, + = 위에서 봐서 반시계."""
    c = base_T_world[:3, :3] @ centre + base_T_world[:3, 3]
    ref = np.arctan2(-c[1], -c[0])
    out = []
    for elev, az in views:
        e, a = np.radians(elev), ref + np.radians(az)
        out.append(base_T_world[:3, :3].T @ [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return out


def view_pose(direction, x_hint, centre, distance):
    """centre 를 보는 카메라 자세(z 광축) — direction 은 물체→카메라, roll 은 x_hint 에 가깝게."""
    z = -np.asarray(direction, float) / np.linalg.norm(direction)
    x = x_hint - (x_hint @ z) * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(z, x), z]
    T[:3, 3] = centre - z * distance
    return T


class Scene:
    """작업 좌표 기준 장면: 보드 코너, 물체·받침대 상자, 손가락(카메라 좌표 점 + 화면 영역), 카메라 내부값."""

    def __init__(self, corners, boxes, fingers, finger_mask, K, size, color_T_depth):
        self.corners, self.boxes = corners, [(np.asarray(c, float), np.asarray(s, float)) for c, s in boxes]
        self.fingers, self.finger_mask = fingers, finger_mask
        self.K, self.size, self.color_T_depth = np.asarray(K, float), size, np.asarray(color_T_depth, float)

    def visible_corners(self, world_T_depth):
        T_c = world_T_depth @ np.linalg.inv(self.color_T_depth)
        inv = np.linalg.inv(T_c)
        p = self.corners @ inv[:3, :3].T + inv[:3, 3]
        uv = p @ self.K.T
        uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-9)
        W, H = self.size
        n = 0
        for k, (u, v) in enumerate(uv):
            if p[k, 2] <= 0 or not (15 < u < W - 15 and 15 < v < H - 15):
                continue
            if self.finger_mask is not None and self.finger_mask[int(v), int(u)]:
                continue
            if any(segment_hits_box(T_c[:3, 3], self.corners[k], c, s) for c, s in self.boxes):
                continue
            n += 1
        return n

    def finger_clearance(self, world_T_depth):
        """(손가락 ↔ 상자들 최소 거리, 손가락 최저 높이) [m] — 손가락 모양이 없으면 (inf, inf)."""
        if self.fingers is None or not len(self.fingers):
            return np.inf, np.inf
        pw = self.fingers @ world_T_depth[:3, :3].T + world_T_depth[:3, 3]
        clear = min(float(box_distance(pw, c, s).min()) for c, s in self.boxes)
        return clear, float(pw[:, 2].min())


def plan_views(ffk, seeds, base_T_world, tool_T_depth, scene, centre, *, distances, raise_deg,
               min_corners, finger_clear, max_move_deg, max_reach, lo, hi, directions=None, turns=1,
               seeds_per_view=4):
    """시드 관절값마다(directions 를 주면 방향마다) → 통과한 가장 가까운 시점(dict) 또는 실패 사유(dict, ok=False).

    directions: 작업 좌표의 시점 방향(물체→카메라) — IK 시드는 카메라 방향이 가까운 seeds_per_view 개.
    turns: 판을 centre 를 지나는 z 축으로 360/turns° 씩 돌린 모든 경우의 최솟값으로 코너·손가락을 본다
    (판을 돌리는 대신 카메라를 반대로 돌려 같은 장면에서 잰다).
    """
    world_T_base = np.linalg.inv(base_T_world)
    cams = [world_T_base @ ffk.pose(q) @ tool_T_depth for q in seeds]      # 작업 좌표의 시드 카메라
    seed_dirs = [(c[:3, 3] - centre) / np.linalg.norm(c[:3, 3] - centre) for c in cams]
    if directions is None:
        jobs = [(d, [i]) for i, d in enumerate(seed_dirs)]
    else:
        jobs = [(d / np.linalg.norm(d), list(np.argsort([-(s @ d) for s in seed_dirs])[:seeds_per_view]))
                for d in map(np.asarray, directions)]
    top = np.radians(85 if directions is None else 90)
    spins = [spin(2 * np.pi * k / turns, centre) for k in range(turns)]
    out = []
    for number, (d0, candidates) in enumerate(jobs, 1):
        az, elev0 = np.arctan2(d0[1], d0[0]), np.arcsin(np.clip(d0[2], -1, 1))
        best = first_fail = None
        for distance in distances:                                      # 가까운 거리부터 — 처음 통과한 것
            for d_elev in raise_deg:
                elev = np.clip(elev0 + np.radians(d_elev), np.radians(5), top)
                direction = np.array([np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)])
                for i in candidates:
                    q0 = seeds[i]
                    T = view_pose(direction, cams[i][:3, 0], centre, distance)
                    q = solve_ik(ffk, base_T_world @ T @ np.linalg.inv(tool_T_depth), [q0], lo, hi)
                    if q is None:
                        continue
                    corners = [scene.visible_corners(S @ T) for S in spins]
                    clear = [scene.finger_clearance(S @ T) for S in spins]
                    entry = {"number": number, "seed": int(i) + 1, "q": q, "distance_m": float(distance),
                             "elev_deg": float(np.degrees(elev)), "seed_elev_deg": float(np.degrees(elev0)),
                             "azimuth_deg": float(np.degrees(az)),
                             "seed_distance_m": float(np.linalg.norm(cams[i][:3, 3] - centre)),
                             "reach_m": wrist_reach(ffk.frames(q)),
                             "move_deg": float(np.degrees(np.max(np.abs(q - q0)))),
                             "corners": min(corners), "corners_by_turn": corners,
                             "finger_clear_m": min(c for c, _ in clear), "finger_low_m": min(z for _, z in clear)}
                    entry["ok"] = (entry["reach_m"] <= max_reach and entry["move_deg"] <= max_move_deg
                                   and entry["corners"] >= min_corners and entry["finger_clear_m"] >= finger_clear
                                   and entry["finger_low_m"] >= finger_clear)
                    if entry["ok"]:
                        best = entry
                        break
                    first_fail = first_fail or entry
                if best:
                    break
            if best:
                break
        out.append(best or first_fail or {"number": number, "ok": False, "reason": "IK 해 없음"})
    return out


def load_seeds(args):
    import yaml
    seeds, names = [], []
    for path in args.views_from or args.poses:
        path = Path(path)
        path = path if path.is_absolute() else ROOT / path
        if args.views_from:
            seeds += [np.asarray(f["q"], float) for f in json.loads((path / "capture.json").read_text())["frames"]]
        else:
            seeds += [np.radians(np.asarray(q, float)) for q in yaml.safe_load(path.read_text())["poses_deg"]]
        names.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    return seeds, ", ".join(names)


def load_work_frame(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    if path.is_dir():
        meta = json.loads(((path / "capture.json") if (path / "capture.json").exists() else (path / "plan.json")).read_text())
        if meta.get("world_frame") != "board" or "base_T_world" not in meta:
            raise SystemExit(f"{path} 는 보드 작업 좌표 run 이 아닙니다")
        return transform(meta["base_T_world"])
    return transform(json.loads(path.read_text())["base_T_world"])


def load_fingers(run, color_size):
    """run 의 depth 에서 카메라 앞 FINGER_RANGE_M 안 점(카메라 좌표)과 화면 영역(팽창)."""
    import cv2
    from core.reconstruction import storage
    run = Path(run)
    frames = storage.load_frames(run if run.is_absolute() else ROOT / run)
    points, mask = [], None
    for fr in frames:
        d, K = fr["depth"], fr["K"]
        m = np.isfinite(d) & (d > FINGER_RANGE_M[0]) & (d < FINGER_RANGE_M[1])
        v, u = np.nonzero(m)
        z = d[m]
        points.append(np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z][::7])
        mask = m if mask is None else mask | m
    if mask.shape[::-1] != tuple(color_size):
        print("⚠️ 손가락 run 의 depth 크기가 RGB 와 달라 화면 가림은 빼고 계산합니다")
        mask = None
    else:
        k = np.ones((FINGER_MASK_DILATE_PX, FINGER_MASK_DILATE_PX), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    return np.concatenate(points), mask


def obstacle_for_tour(base_T_world, boxes, margin=0.01, turns=1, axis=(0, 0)):
    """pose_tour --obstacle 용 base 기준 축 정렬 상자 — 상자들 꼭짓점을 감싸고 margin 만큼 키운다.

    turns > 1: 상자들을 axis(작업 좌표 x, y)를 지나는 z 축으로 360/turns° 씩 돌린 것까지 감싼다 (회전판).
    """
    corners = []
    for k in range(turns):
        S = spin(2 * np.pi * k / turns, axis)
        for c, s in boxes:
            for sign in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
                p = S[:3, :3] @ (np.asarray(c) + sign * np.asarray(s) / 2) + S[:3, 3]
                corners.append(base_T_world[:3, :3] @ p + base_T_world[:3, 3])
    corners = np.array(corners)
    lo, hi = corners.min(axis=0) - margin, corners.max(axis=0) + margin
    lo[2] = max(lo[2], 0.0)
    return np.r_[(lo + hi) / 2, hi - lo]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--views-from", nargs="+", help="시점 방향·IK 시드: 촬영 run 폴더 (capture.json 관절값)")
    src.add_argument("--poses", nargs="+", help="시점 방향·IK 시드: 자세 YAML (poses_deg)")
    p.add_argument("--view", type=float, nargs=2, action="append", metavar=("ELEV", "AZ"),
                   help="시점 방향을 직접: 높이각·방위 [deg] (방위 0 = 물체에서 로봇 쪽, + = 위에서 봐서 반시계, "
                        "base 기준). 주면 시드는 IK 출발점으로만 쓴다")
    p.add_argument("--turns", type=int, default=1,
                   help="회전판: 판을 물체 중심 수직축으로 360/N° 씩 돌린 N 경우 모두에서 코너·손가락 검사")
    p.add_argument("--work-frame", required=True,
                   help="작업 좌표의 base 위치: 보드 작업 좌표 run 폴더 또는 board_check.py --save-frame JSON")
    p.add_argument("--object", type=float, nargs=6, required=True, metavar=("CX", "CY", "CZ", "SX", "SY", "SZ"),
                   help="물체 상자 — 작업 좌표의 중심·전체 크기 [m] (카메라는 이 중심을 본다)")
    p.add_argument("--keep-out", type=float, nargs=6, action="append", default=[],
                   metavar=("CX", "CY", "CZ", "SX", "SY", "SZ"), help="손가락이 피하고 코너를 가리는 상자(받침대 등)")
    p.add_argument("--fingers-from", help="손가락이 찍힌 run 폴더(벌린 그리퍼) — 없으면 손가락 검사를 건너뛴다")
    p.add_argument("--distances", type=float, nargs="+", default=[0.16, 0.17, 0.18, 0.19, 0.20, 0.22],
                   help="물체 중심 ↔ depth 카메라 거리 후보 [m] — 앞의 것부터")
    p.add_argument("--raise-deg", type=float, nargs="+",
                   help="시점 높이각에 더해 볼 값 [deg] — 앞의 것부터 (기본: 시드 방향 -10 0 8 15 20, --view 0 5 10)")
    p.add_argument("--min-corners", type=int, default=26, help="계산상 보이는 보드 코너 최소 (실제 ~55–60%%)")
    p.add_argument("--finger-clear", type=float, default=0.025, help="손가락 ↔ 상자·보드 최소 거리 [m]")
    p.add_argument("--max-move-deg", type=float, default=60, help="시드 관절값에서 최대 변화 [deg]")
    p.add_argument("--max-reach", type=float, default=0.30, help="joint2 축 ↔ 손목 수평 거리 상한 [m]")
    p.add_argument("--out", required=True, help="pose_tour --poses YAML")
    p.add_argument("--config", default=str(ROOT / "workcell/scans/piper_d405.yaml"))
    p.add_argument("--calibration", default=str(ROOT / "workcell/calibration/piper_d405.yaml"))
    args = p.parse_args()
    if args.turns < 1:
        p.error("--turns 는 1 이상")
    if args.raise_deg is None:
        args.raise_deg = [0, 5, 10] if args.view else [-10, 0, 8, 15, 20]

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    cal = yaml.safe_load(Path(args.calibration).read_text())
    ffk = FastFK(PiperFK(ROOT / cfg["urdf"], cfg["base_frame"], cfg["tool_frame"]))
    lo = ffk.lower + np.radians(LIMIT_MARGIN_DEG)
    hi = ffk.upper - np.radians(LIMIT_MARGIN_DEG)
    tool_T_depth = transform(cal["tool_T_depth"])
    color = cal["identity"]["color"]
    size = (int(color["width"]), int(color["height"]))
    base_T_world = load_work_frame(args.work_frame)
    seeds, source = load_seeds(args)
    boxes = [(args.object[:3], args.object[3:])] + [(b[:3], b[3:]) for b in args.keep_out]
    fingers = mask = None
    if args.fingers_from:
        fingers, mask = load_fingers(args.fingers_from, size)
        print(f"손가락: 점 {len(fingers)}개, 카메라 앞 {fingers[:, 2].min() * 100:.1f}–{fingers[:, 2].max() * 100:.1f} cm"
              + (f", 화면 가림 {mask.mean():.0%}" if mask is not None else ""))
    else:
        print("⚠️ --fingers-from 이 없어 손가락 충돌·가림을 검사하지 않습니다")
    scene = Scene(board_corners(cfg["scan_board"]), boxes, fingers, mask,
                  np.asarray(color["K"], float).reshape(3, 3), size, cal["identity"]["color_T_depth"])
    centre = np.asarray(args.object[:3])
    directions = view_directions(args.view, base_T_world, centre) if args.view else None
    rows = plan_views(ffk, seeds, base_T_world, tool_T_depth, scene, centre,
                      distances=args.distances, raise_deg=args.raise_deg, min_corners=args.min_corners,
                      finger_clear=args.finger_clear, max_move_deg=args.max_move_deg, max_reach=args.max_reach,
                      lo=lo, hi=hi, directions=directions, turns=args.turns)
    kept = []
    for r in rows:
        if "q" not in r:
            print(f"#{r['number']:2d} {r['reason']}")
            continue
        view = (f"시점 높이각 {args.view[r['number'] - 1][0]:3.0f}° 방위 {args.view[r['number'] - 1][1]:+4.0f}° → "
                f"높이각 {r['elev_deg']:3.0f}° (시드 #{r['seed']})" if args.view else
                f"방위 {r['azimuth_deg']:4.0f}° 높이각 {r['seed_elev_deg']:3.0f}→{r['elev_deg']:3.0f}°")
        corners = "/".join(map(str, r["corners_by_turn"])) if args.turns > 1 else f"{r['corners']:2d}"
        print(f"#{r['number']:2d} {view} | "
              f"{r['seed_distance_m'] * 100:.0f}→{r['distance_m'] * 100:.0f} cm {'통과' if r['ok'] else '실패'} | "
              f"코너 {corners} | 손가락-상자 {r['finger_clear_m'] * 100:4.1f} cm, 최저 {r['finger_low_m'] * 100:4.1f} cm | "
              f"뻗음 {r['reach_m'] * 100:.1f} cm | 관절 변화 {r['move_deg']:.0f}°")
        if r["ok"]:
            kept.append(r)
    obstacle = obstacle_for_tour(base_T_world, boxes, turns=args.turns, axis=centre[:2])
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# scan_poses.py — 시점 {source}, 작업 좌표 {args.work_frame}",
             f"# 물체 {args.object}, 받침대 등 {args.keep_out}"
             + (f", 판 {360 / args.turns:g}° 씩 {args.turns}번 돌림" if args.turns > 1 else ""),
             f"# pose_tour 충돌 상자(base): --obstacle {' '.join(f'{v:.3f}' for v in obstacle)}",
             "# 각 줄: 관절값[deg] — 주석: 원래 시점 번호·물체 중심까지 거리·높이각·계산 코너·손가락 여유",
             "poses_deg:"]
    for r in kept:
        lines.append(f"- {yaml.safe_dump([round(float(x), 3) for x in np.degrees(r['q'])], default_flow_style=True).strip()}"
                     f"  # #{r['number']} {r['distance_m'] * 100:.0f}cm 높이각 {r['elev_deg']:.0f}°"
                     + (f" 방위 {args.view[r['number'] - 1][1]:+.0f}°" if args.view else "") + f" 코너 {r['corners']} "
                     f"손가락 {r['finger_clear_m'] * 100:.1f}cm")
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(kept)}/{len(rows)}개 → {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    print(f"pose_tour: --poses {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} "
          f"--obstacle {' '.join(f'{v:.3f}' for v in obstacle)}")


if __name__ == "__main__":
    main()

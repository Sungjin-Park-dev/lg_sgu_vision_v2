#!/usr/bin/env python3
"""스캔 궤적의 **한 점**을 중심으로 카메라를 기울이는(tilt) 검사 모션을 CSV(+npz)로 낸다.

스캔 궤적(`glns/solve.py` → `glns/verify.py`)이 "모든 viewpoint 를 한 번씩" 이라면, 이건
그 반대다 — **한 자리**를 표면점 중심으로 공전(orbit)하며 여러 각도에서 본다.

중심은 궤적 CSV 의 행으로 고른다. CSV 가 각 행의 카메라 포즈를 이미 갖고 있어서(FK 산출)
viewpoint 뿐 아니라 **보간으로 생긴 점, 모션플래너가 만든 점**도 중심이 될 수 있다 —
viewpoint h5 에는 그런 점이 아예 없다. 각 행의 출처는 CSV 의 ``waypoint_kind`` 컬럼
(viewpoint / interpolated / planned)에 있고, 이 스크립트가 고른 행의 라벨을 찍어 준다.
정면에서 안 보이던 결함(광택면 반사, 단차, 그림자)을 각도로 잡아내는 용도라 순서 최적화
(GTSP/GLNS)가 필요 없다: 방문 순서가 처음부터 정해져 있다.

    중심 → 상 → 중심 → 하 → 중심 → 좌 → 중심 → 우 → 중심

각 leg 는 카메라를 표면점 둘레로 공전시킨다 — **작업거리(WD)와 주시점은 그대로**이고 시선
방향만 기운다. 그래서 tilt 중에도 대상이 항상 화면 중앙, 초점거리 안에 있다.

새 알고리즘은 없다. 기존 조각의 조립이다:
    build_camera_poses → orbit 포즈 생성 → solve_ik_multi_seed → 충돌 필터
    → 분기 선택 DP(Viterbi) → 2π unwrap → collision_gate_and_save

도달 불가 각도 처리: leg 별로 **중심에서부터 연속으로 IK/충돌이 통과하는 만큼만** 왕복한다
(예: ±20° 요청이 12° 에서 막히면 그 leg 만 ±12°). 축소된 leg 는 로그에 크게 남기고 npz meta
에도 기록한다. `--no-clamp` 를 주면 축소 대신 실패한다.

물체 pose 는 `--object-position/--object-quat` 로 받아 `config.TARGET_OBJECT` 에 덮어쓴다 —
`glns/solve.py` · `plan_move.py` 와 같은 관례다(호출자가 살아있는 기즈모 pose 를 넘긴다).

Exit: 0 = 충돌-free CSV 생성, 2 = 중심 포즈 도달 불가 / 충돌 게이트 실패 / 인자 오류.

Usage:
    uv run --no-sync scripts/core/trajectory/tilt_motion.py \\
        --object sample --trajectory data/sample/trajectory/74/trajectory.csv \\
        --row-index 120 \\
        --object-position -0.15 0.741 0.19 --object-quat 0.70710678 0 0 0.70710678 \\
        --pitch-min -20 --pitch-max 20 --pitch-n 40 \\
        --roll-min -20 --roll-max 20 --roll-n 40 \\
        --output data/sample/trajectory/74/trajectory_tilt_vp0010.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config  # noqa: E402
from common.math_utils import quaternion_to_rotation_matrix  # noqa: E402
from common.tilt_geometry import camera_pose, tilt_legs  # noqa: E402
from core import trajectory as PT  # noqa: E402
from core.glns.candidates import (  # noqa: E402
    _collision_filter_representatives,
    _joint_limits_and_periods,
)
from core.trajectory.periodic import periodic_joint_delta  # noqa: E402

# 분기 전환(같은 포즈의 다른 팔 자세) 억제용 DP 페널티. 인접 waypoint 사이 L∞ 가
# reconfig 임계를 넘으면 붙는다 - tilt 는 연속 모션이라 분기가 갈리면 안 된다.
BRANCH_SWITCH_PENALTY = 1e3


def _joints(text: str) -> np.ndarray:
    """argparse ``type=`` — "q0,...,q5" → (6,) ndarray (plan_move.py 와 동일 규약)."""
    values = [v for v in text.replace(",", " ").split() if v]
    try:
        q = np.array([float(v) for v in values], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"could not parse as numbers ({exc})")
    if q.shape != (6,):
        raise argparse.ArgumentTypeError(f"expected 6 joint values, got {len(values)}")
    return q


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True, help="object name placed in the collision world")
    p.add_argument("--trajectory", required=True, type=Path,
                   help="scan trajectory CSV whose row is the tilt centre")
    p.add_argument("--row-index", required=True, type=int,
                   help="row of that CSV to tilt around (0-based). viewpoint 행뿐 아니라 "
                        "보간·모션플래닝으로 생긴 행도 고를 수 있다")
    p.add_argument("--output", required=True, type=Path, help="output CSV path")
    p.add_argument("--object-position", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   default=None, help="object position (robot base_link frame, m)")
    p.add_argument("--object-quat", type=float, nargs=4, metavar=("W", "X", "Y", "Z"),
                   default=None, help="object orientation quaternion")
    # 각도 범위. n 은 '중심 → 끝' 한쪽 방향의 샘플 수(중심 포함)이므로 leg 당 새 포즈는 n-1 개다.
    p.add_argument("--pitch-min", type=float, default=-20.0, help="downward tilt angle [deg]")
    p.add_argument("--pitch-max", type=float, default=20.0, help="upward tilt angle [deg]")
    p.add_argument("--pitch-n", type=int, default=40, help="pitch samples per side (default 40)")
    p.add_argument("--roll-min", type=float, default=-20.0, help="left tilt angle [deg]")
    p.add_argument("--roll-max", type=float, default=20.0, help="right tilt angle [deg]")
    p.add_argument("--roll-n", type=int, default=40, help="roll samples per side (default 40)")
    # IK
    p.add_argument("--num-seeds", type=int, default=32,
                   help="IK seeds per pose (default 32)")
    p.add_argument("--batch-size", type=int, default=128, help="IK batch size (default 128)")
    p.add_argument("--ik-seed", type=int, default=PT.IK_RANDOM_SEED,
                   help=f"deterministic cuRobo IK seed (default {PT.IK_RANDOM_SEED})")
    p.add_argument("--dedup-rad", type=float, default=PT.CANDIDATE_DEDUP_RAD,
                   help=f"near-duplicate L-inf threshold for IK candidates in rad (default {PT.CANDIDATE_DEDUP_RAD})")
    # 첫 값이 음수면 '-1.5,...' 로 시작해 argparse 가 옵션으로 오인한다 — 호출자는
    # --anchor-joints=<v> 형태(공백 아님)로 넘길 것.
    p.add_argument("--anchor-joints", type=_joints, default=None,
                   help='reference pose "q0,...,q5" [rad] for branch selection; normally the current robot pose')
    p.add_argument("--no-clamp", dest="clamp", action="store_false",
                   help="fail instead of clamping when an angle is unreachable")
    p.set_defaults(clamp=True)
    PT.add_timing_cli_arguments(p)
    args = p.parse_args()

    if args.row_index < 0:
        p.error("--row-index must be >= 0")
    for name in ("pitch_n", "roll_n"):
        if getattr(args, name) < 2:
            p.error(f"--{name.replace('_', '-')} must be >= 2")
    if args.pitch_max < 0.0 or args.roll_max < 0.0:
        p.error("--pitch-max / --roll-max must be >= 0 (the opposite direction is min)")
    if args.pitch_min > 0.0 or args.roll_min > 0.0:
        p.error("--pitch-min / --roll-min must be <= 0 (the same direction is max)")
    if args.num_seeds <= 0 or args.batch_size <= 0:
        p.error("--num-seeds / --batch-size must be > 0")
    return args



def load_trajectory_row(csv_path: Path, row_index: int):
    """궤적 CSV 의 한 행 → (카메라 4x4 포즈, 라벨, 총 행 수).

    CSV 는 이미 각 행의 카메라 포즈를 갖고 있다(target-POS_* / target-ROT_*, FK 산출).
    그래서 viewpoint h5 를 다시 읽어 자세를 재구성할 필요가 없고, **보간·모션플래닝으로
    생긴 행도 중심으로 삼을 수 있다** — viewpoint h5 에는 그런 점이 아예 없다.

    라벨(waypoint_kind)은 있으면 읽고 없으면 "unknown". 옛 CSV 에는 그 컬럼이 없다.
    """
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"trajectory CSV has no rows: {csv_path}")
    if not 0 <= row_index < len(rows):
        raise IndexError(f"row {row_index} out of range (CSV has {len(rows)} rows)")
    row = rows[row_index]
    need = ("target-POS_X", "target-POS_Y", "target-POS_Z",
            "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W")
    missing = [c for c in need if c not in row]
    if missing:
        raise ValueError(f"trajectory CSV lacks camera pose columns: {missing}")
    pos = np.array([float(row[c]) for c in need[:3]], dtype=np.float64)
    qx, qy, qz, qw = (float(row[c]) for c in need[3:])
    pose = np.eye(4, dtype=np.float64)
    # CSV 는 (x,y,z,w), quaternion_to_rotation_matrix 는 (w,x,y,z) 를 받는다.
    pose[:3, :3] = quaternion_to_rotation_matrix(np.array([qw, qx, qy, qz]))
    pose[:3, 3] = pos
    return pose, (row.get("waypoint_kind") or "unknown"), len(rows)


def load_trajectory_meta(csv_path: Path) -> dict:
    """CSV 옆 npz sidecar 의 meta. 작업거리와 물체 배치가 거기 박혀 있다."""
    npz = Path(csv_path).with_suffix(".npz")
    if not npz.exists():
        return {}
    try:
        with np.load(npz, allow_pickle=False) as d:
            return json.loads(str(d["meta"])) if "meta" in d else {}
    except (ValueError, TypeError, KeyError):
        return {}


# =============================================================================
# IK 후보 + 분기 선택
# =============================================================================

def _solve_candidates(poses, robot_cfg, world, *, num_seeds, batch_size, ik_seed,
                      dedup_rad, joint_periods):
    """포즈마다 collision-free IK 후보 집합을 만든다 → list[(K_i, 6)]."""
    positions = np.ascontiguousarray(poses[:, :3, 3])
    quats = PT.rot_to_quat_batch(poses[:, :3, :3])
    sols, success = PT.solve_ik_multi_seed(
        robot_cfg, world, positions, quats,
        num_seeds=num_seeds, batch_size=batch_size, random_seed=ik_seed,
    )

    representatives = []
    for i in range(len(poses)):
        kept: list[np.ndarray] = []
        for q in sols[i][success[i]]:
            q = np.asarray(q, dtype=np.float64)
            if any(float(np.max(np.abs(periodic_joint_delta(q - prior, joint_periods))))
                   <= dedup_rad for prior in kept):
                continue
            kept.append(q)
        representatives.append(np.asarray(kept, dtype=np.float64).reshape(-1, 6))

    removed = _collision_filter_representatives(representatives, robot_cfg, world)
    total = sum(len(r) for r in representatives)
    print(f"  {total} IK candidates over {len(poses)} poses "
          f"({removed} removed by collision)")
    return representatives


def _chain_dp(candidate_sequence, joint_periods, reconfig_rad, anchor=None):
    """순서가 고정된 포즈열에서 waypoint 마다 IK 분기 하나를 고른다 (Viterbi).

    비용 = 인접 관절 이동량(주기 보정 L2) + 분기 전환 페널티. anchor 를 주면 첫 waypoint 를
    그 자세에 가까운 분기로 시작한다 — 로봇이 지금 있는 자리에서 가장 짧게 진입한다.
    """
    prev = candidate_sequence[0]
    if anchor is None:
        costs = np.zeros(len(prev), dtype=np.float64)
    else:
        costs = np.linalg.norm(
            periodic_joint_delta(prev - np.asarray(anchor, dtype=np.float64), joint_periods),
            axis=1,
        )

    parents = []
    for step in range(1, len(candidate_sequence)):
        cur = candidate_sequence[step]
        delta = periodic_joint_delta(cur[None, :, :] - prev[:, None, :], joint_periods)
        edge = np.linalg.norm(delta, axis=2)
        edge = edge + np.where(np.max(np.abs(delta), axis=2) > reconfig_rad,
                               BRANCH_SWITCH_PENALTY, 0.0)
        total = costs[:, None] + edge
        best = np.argmin(total, axis=0)
        costs = total[best, np.arange(len(cur))]
        parents.append(best)
        prev = cur

    idx = int(np.argmin(costs))
    chosen = [idx]
    for best in reversed(parents):
        idx = int(best[idx])
        chosen.append(idx)
    chosen.reverse()
    traj = np.stack([candidate_sequence[t][chosen[t]] for t in range(len(chosen))])
    return traj, float(np.min(costs))


def _unwrap_chain(traj, joint_lower, joint_upper, joint_periods, anchor=None):
    """이웃 waypoint 와 2π 이상 벌어지지 않도록 주기 관절값을 관절한계 안에서 다시 고른다.

    IK 해는 [-π, π] 로 정규화돼 나오므로, 연속 모션이어도 π 를 넘는 순간 값이 튄다. 실행기는
    waypoint 사이를 관절공간 직선으로 채우므로 그대로 두면 손목이 한 바퀴 돈다.
    """
    out = np.asarray(traj, dtype=np.float64).copy()
    periodic = np.flatnonzero(np.asarray(joint_periods) > 0.0)
    if not len(periodic):
        return out
    for t in range(len(out)):
        reference = out[t - 1] if t > 0 else (anchor if anchor is not None else out[0])
        for j in periodic:
            period = float(joint_periods[j])
            k = float(np.round((float(reference[j]) - out[t, j]) / period))
            value = out[t, j] + k * period
            while value > joint_upper[j] + 1e-9:
                value -= period
            while value < joint_lower[j] - 1e-9:
                value += period
            out[t, j] = value
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    print("=" * 64)
    print("TILT MOTION (orbit around one trajectory row)")
    print("=" * 64)

    # 충돌 게이트가 시간을 매기기 전에만 반영되면 된다. 여기가 그 앞이다.
    PT.apply_timing_cli(args)

    # 물체 배치: 기본값 → CLI override (plan_move.py / solve.py 와 같은 관례).
    config.apply_object_placement(args.object)
    if args.object_position is not None:
        config.TARGET_OBJECT["position"] = np.array(args.object_position, dtype=np.float64)
    if args.object_quat is not None:
        config.TARGET_OBJECT["rotation"] = np.array(args.object_quat, dtype=np.float64)

    # 궤적 옆 npz meta 가 작업거리와 물체 배치를 갖고 있다 — 그 궤적이 계획된 바로 그 조건.
    # CLI override 가 있으면 그쪽이 이긴다(호출자가 살아있는 기즈모 pose 를 줄 수 있다).
    meta = load_trajectory_meta(args.trajectory)
    if args.object_position is None and "object_position" in meta:
        config.TARGET_OBJECT["position"] = np.asarray(meta["object_position"], dtype=np.float64)
    if args.object_quat is None and "object_quat_wxyz" in meta:
        config.TARGET_OBJECT["rotation"] = np.asarray(meta["object_quat_wxyz"], dtype=np.float64)
    wd_m = float(meta.get("working_distance_mm", config.CAMERA_WORKING_DISTANCE_MM)) / 1000.0

    try:
        center_pose, row_kind, n_rows = load_trajectory_row(args.trajectory, args.row_index)
    except (ValueError, IndexError) as exc:
        print(f"  x {exc}")
        return 2

    print(f"  object    : {args.object}  pos="
          f"{np.round(config.TARGET_OBJECT['position'], 3).tolist()} "
          f"quat={np.round(config.TARGET_OBJECT['rotation'], 4).tolist()}")
    print(f"  trajectory: {args.trajectory} ({n_rows} rows, WD={wd_m * 1000:.1f} mm)")
    print(f"  center    : row #{args.row_index}  [{row_kind}]")
    print(f"  pitch up/down : {args.pitch_max:+.1f} / {args.pitch_min:+.1f} deg  x{args.pitch_n}")
    print(f"  roll left/right: {args.roll_min:+.1f} / {args.roll_max:+.1f} deg  x{args.roll_n}")
    print(f"  camera    : pos={np.round(center_pose[:3, 3], 4).tolist()} -> "
          f"surface={np.round(center_pose[:3, 3] + center_pose[:3, 2] * wd_m, 4).tolist()}")

    # ---- 포즈 생성: 중심 1개 + leg 별 (n-1)개 -------------------------------
    # 기하는 common.tilt_geometry 한 벌뿐이다 — Isaac UI 의 부채꼴 미리보기가 같은 함수를
    # 쓰므로, 화면에 보이는 부채꼴과 여기서 IK 를 푸는 포즈가 어긋날 수 없다.
    target, raw_legs = tilt_legs(
        center_pose, wd_m,
        pitch_min=args.pitch_min, pitch_max=args.pitch_max, pitch_n=args.pitch_n,
        roll_min=args.roll_min, roll_max=args.roll_max, roll_n=args.roll_n)
    poses = [center_pose]
    legs = []                                  # (라벨, [pose_idx...], [angle...])
    for label, leg_poses, leg_angles in raw_legs:
        start = len(poses)
        poses.extend(leg_poses)
        legs.append((label, list(range(start, start + len(leg_poses))), leg_angles))
    if not legs:
        print("  x every leg angle is 0 deg - there is no motion to build.")
        return 2
    poses = np.stack(poses)
    print(f"  {len(poses)} poses (1 center + {len(legs)} legs)")

    # ---- IK + 충돌 필터 ----------------------------------------------------
    print("-" * 64)
    print("IK (multi-seed) + collision filter")
    robot_cfg = PT.resolve_robot_config(PT.ROBOT_CONFIG)
    world_config = PT.build_collision_world(args.object)
    joint_lower, joint_upper, joint_periods = _joint_limits_and_periods(robot_cfg)
    candidates = _solve_candidates(
        poses, robot_cfg, world_config,
        num_seeds=args.num_seeds, batch_size=args.batch_size, ik_seed=args.ik_seed,
        dedup_rad=args.dedup_rad, joint_periods=joint_periods,
    )

    if not len(candidates[0]):
        print(f"  x no collision-free IK solution for the center pose "
              f"(row #{args.row_index}) - cannot build a tilt here.")
        return 2

    # ---- leg 별 도달 가능 구간까지만 왕복 (요청 각도 축소) ------------------
    print("-" * 64)
    print("reachable angle per leg")
    sequence = [0]                              # 중심에서 시작
    clamped = []
    for label, pose_indices, angles in legs:
        reach = 0
        for pose_idx in pose_indices:
            if not len(candidates[pose_idx]):
                break
            reach += 1
        requested = float(angles[-1])
        if reach == 0:
            print(f"  {label}: requested {requested:+.1f} deg -> 0.0 deg "
                  "(unreachable from the first sample)")
        else:
            print(f"  {label}: requested {requested:+.1f} deg -> "
                  f"{float(angles[reach - 1]):+.1f} deg "
                  f"({reach}/{len(pose_indices)} samples)")
        if reach < len(pose_indices):
            reached = float(angles[reach - 1]) if reach else 0.0
            clamped.append({"leg": label, "requested_deg": requested,
                            "reached_deg": reached, "samples": int(reach)})
            if not args.clamp:
                print(f"  x --no-clamp: leg {label} cannot reach {requested:+.1f} deg.")
                return 2
        if reach == 0:
            continue                            # 이 방향으로는 한 샘플도 못 간다
        used = pose_indices[:reach]
        sequence += used + used[-2::-1] + [0]   # 중심 → 끝 → 중심 (끝 포즈는 한 번만)

    if clamped:
        print()
        print("  ! some legs could not reach the requested angle "
              "(unreachable/collision -> clamped):")
        for item in clamped:
            print(f"      {item['leg']}: {item['requested_deg']:+.1f} -> "
                  f"{item['reached_deg']:+.1f} deg")
        print("    narrow the range, move the object closer to the robot, "
              "or pick another row.")

    if len(sequence) < 2:
        print("  x only the center pose is reachable - no tilt span to move through.")
        return 2

    # ---- 분기 선택 DP → 2π unwrap ------------------------------------------
    print("-" * 64)
    print(f"branch-selection DP: {len(sequence)} waypoints")
    traj, cost = _chain_dp(
        [candidates[i] for i in sequence], joint_periods,
        np.deg2rad(PT.RECONFIG_THRESHOLD_DEG), anchor=args.anchor_joints,
    )
    traj = _unwrap_chain(traj, joint_lower, joint_upper, joint_periods,
                         anchor=args.anchor_joints)
    max_step = float(np.max(np.abs(np.diff(traj, axis=0)))) if len(traj) > 1 else 0.0
    print(f"  path cost {cost:.3f}, max joint step between waypoints "
          f"{np.rad2deg(max_step):.2f} deg")
    if cost >= BRANCH_SWITCH_PENALTY:
        print("  ! the path switches arm branch midway - check it in preview first.")

    # ---- 충돌 게이트 + 저장 -------------------------------------------------
    print("-" * 64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 전 구간이 검사 모션이다(transit 아님) → EE 선속도/각속도 기준으로 시간을 매긴다.
    is_transit = np.zeros(len(traj), dtype=bool)
    gate = PT.collision_gate_and_save(
        traj, is_transit, robot_cfg=robot_cfg, world_config=world_config,
        out_csv=args.output,
        # 모든 행이 **명령한 orbit 포즈의 IK 해**다. 여기는 resample 이 없어서 행과 포즈가
        # 1:1 이고, 보간점도 플래너가 만든 점도 섞이지 않는다 → 전부 viewpoint.
        # (안 주면 tilt CSV 만 라벨 없는 파일이 되어, 그걸 다시 tilt 기준으로 골랐을 때
        #  패널이 unknown 만 띄운다.)
        kinds=np.full(len(traj), PT.WAYPOINT_VIEWPOINT, dtype=np.int8),
        meta={
            "kind": "tilt",
            "trajectory": str(args.trajectory),
            "row_index": int(args.row_index),
            "row_kind": row_kind,
            "working_distance_m": float(wd_m),
            "pitch_deg": [float(args.pitch_min), float(args.pitch_max)],
            "roll_deg": [float(args.roll_min), float(args.roll_max)],
            "samples": {"pitch": int(args.pitch_n), "roll": int(args.roll_n)},
            "clamped_legs": clamped,
            "surface_point": np.asarray(target, dtype=float).tolist(),
            "object_position": config.TARGET_OBJECT["position"].astype(float).tolist(),
            "object_quat_wxyz": config.TARGET_OBJECT["rotation"].astype(float).tolist(),
        },
    )
    if not gate["collision_free"]:
        print(f"  x collision gate failed: {gate['n_collisions']} collisions - not saved.")
        print("    (per-pose IK was collision-free, but the interpolation between "
              "waypoints grazes the object. Raise the sample count or "
              "narrow the angles.)")
        return 2

    print(f"  OK waypoints={gate['n_waypoints']}, duration={gate['total_time']:.2f}s")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Check per-viewpoint IK reachability for the current object pose, and save the IK.

This is the Isaac UI's placement-feedback front end for the GLNS candidate stage.
It runs the SAME pipeline glns/solve.py feeds into GTSP — pose augmentation
(roll/tilt) → multi-seed cuRobo IK → near-duplicate dedup → collision filter — and

  1. writes a JSON with one candidate count per viewpoint (HDF5 order) so the UI
     can color the displayed points green/red, and
  2. saves the solved IK candidates (joints + variant metadata) to an HDF5 keyed
     by the object placement and the augmentation/dedup settings they were solved
     with, so Generate Trajectory (solve.py) can reuse them instead of re-solving.

Augmentation/dedup are opt-in flags; the save path encodes them under
``data/{object}/ik/{N}/<nominal|roll|tilt|rolltilt>[_nodedup].h5`` so modes coexist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_ROOT))

from core.trajectory import (  # noqa: E402
    CANDIDATE_DEDUP_RAD,
    IK_BATCH_SIZE,
    IK_RANDOM_SEED,
    NUM_IK_SEEDS,
    ROBOT_CONFIG,
    resolve_robot_config,
    build_camera_poses,
    build_collision_world,
)
from core.glns.candidates import (  # noqa: E402
    _build_pose_variants,
    _collision_filter_representatives,
    _joint_limits_and_periods,
    _solve_pose_variant_candidates,
)
from core.glns.ik_store import (  # noqa: E402
    build_settings,
    ik_solutions_path,
    save_ik_solutions,
)
from core.viewpoint import load_viewpoints_hdf5  # noqa: E402
from common import config, scene_config  # noqa: E402


def _parse_joints(text):
    """"q0,...,q5" → (6,) ndarray. None/빈 문자열이면 None (정렬 안 함)."""
    if not text:
        return None
    values = [v for v in str(text).replace(",", " ").split() if v]
    q = np.asarray([float(v) for v in values], dtype=np.float64)
    if q.shape != (6,):
        raise SystemExit(f"--reference-joints needs 6 values, got {len(values)}")
    return q


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check per-viewpoint IK reachability and save the IK candidates",
    )
    parser.add_argument("--object", type=str, required=True, help="Object name")
    parser.add_argument("--num-viewpoints", type=int, default=None,
                        help="Number of viewpoints, only used if --viewpoints is omitted")
    parser.add_argument("--viewpoints", type=str, default=None,
                        help="Direct path to viewpoints.h5")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path (per-viewpoint candidate counts, for UI coloring)")
    parser.add_argument("--save-ik", type=str, default=None,
                        help="HDF5 path for the saved IK candidates (default: derived from "
                             "the viewpoints path + augmentation/dedup options)")
    parser.add_argument("--num-seeds", type=int, default=NUM_IK_SEEDS,
                        help=f"IK seeds per viewpoint (default: {NUM_IK_SEEDS})")
    parser.add_argument("--batch-size", type=int, default=IK_BATCH_SIZE,
                        help=f"IK batch size (default: {IK_BATCH_SIZE})")
    parser.add_argument("--ik-seed", type=int, default=IK_RANDOM_SEED,
                        help=f"Deterministic cuRobo IK seed (default: {IK_RANDOM_SEED})")
    # Augmentation (matches glns/solve.py) — off by default (nominal only).
    parser.add_argument("--roll-augment", action="store_true",
                        help="add optical-axis roll IK pose variants")
    parser.add_argument("--roll-step-deg", type=float, default=30.0,
                        help="[--roll-augment] roll sweep step deg (default: 30 → 11 poses)")
    parser.add_argument("--tilt-augment", action="store_true",
                        help="add off-normal tilt IK pose variants")
    parser.add_argument("--tilt-angles-deg", type=float, nargs="+", default=[5.0, 10.0],
                        help="[--tilt-augment] tilt magnitudes (default: 5 10)")
    parser.add_argument("--tilt-azimuths", type=int, default=8,
                        help="[--tilt-augment] evenly spaced tilt azimuths (default: 8)")
    # Dedup — on by default, matching glns/solve.py.
    parser.add_argument("--no-dedup", dest="dedup", action="store_false",
                        help="keep all IK solutions instead of removing near-duplicates")
    parser.set_defaults(dedup=True)
    parser.add_argument("--dedup-rad", type=float, default=CANDIDATE_DEDUP_RAD,
                        help=f"[dedup] L-inf joint threshold rad (default: {CANDIDATE_DEDUP_RAD})")
    scene_config.add_cli_argument(parser)
    # 첫 값이 음수면 argparse 가 옵션으로 오인한다 — 호출자는 --reference-joints=<v> 형태로.
    parser.add_argument("--reference-joints", type=str, default=None,
                        help='reference joints "q0,...,q5" [rad] — IK 해를 이 자세에 가장 가까운 2π 등가로 옮긴다(자세는 그대로, 회전량만 줄어든다). 보통 로봇의 현재 관절값을 준다')
    parser.add_argument("--object-position", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Override target object position in robot-base frame (meters)")
    parser.add_argument("--object-quat", type=float, nargs=4, default=None,
                        metavar=("W", "X", "Y", "Z"),
                        help="Override target object orientation quaternion (w x y z)")
    args = parser.parse_args()

    if args.num_seeds <= 0:
        parser.error("--num-seeds must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if not 0.0 < args.roll_step_deg <= 180.0:
        parser.error("--roll-step-deg must be in (0, 180]")
    if args.tilt_azimuths < 1:
        parser.error("--tilt-azimuths must be >= 1")
    if any(a <= 0.0 or a > 45.0 for a in args.tilt_angles_deg):
        parser.error("--tilt-angles-deg values must be in (0, 45]")
    if args.dedup and args.dedup_rad <= 0.0:
        parser.error("--dedup-rad must be > 0")
    if args.viewpoints is None and args.num_viewpoints is None:
        parser.error("Either --viewpoints or --num-viewpoints is required")
    return args


def main() -> None:
    args = _parse_args()

    # 씬을 먼저 반영한다 — 물체 배치(object_placements)가 이제 씬 소유다.
    scene_config.apply_cli(args, config)
    if config.apply_object_placement(args.object):
        print(f"  Per-object placement '{args.object}': pos={config.TARGET_OBJECT['position']}, "
              f"quat={config.TARGET_OBJECT['rotation']}")
    if args.object_position is not None:
        config.TARGET_OBJECT["position"] = np.array(args.object_position, dtype=np.float64)
        print(f"  Object position override (robot frame): {args.object_position}")
    if args.object_quat is not None:
        config.TARGET_OBJECT["rotation"] = np.array(args.object_quat, dtype=np.float64)
        print(f"  Object rotation override (w,x,y,z): {args.object_quat}")

    h5_path = (
        Path(args.viewpoints)
        if args.viewpoints
        else config.resolve_viewpoint_path(args.object, args.num_viewpoints)
    )

    print("[1/5] Loading viewpoints...")
    viewpoint = load_viewpoints_hdf5(h5_path)
    positions = viewpoint.positions
    normals = viewpoint.normals
    n_viewpoints = len(positions)
    wd_m = viewpoint.working_distance_m
    print(f"  Loaded from {h5_path}")
    print(f"  {n_viewpoints} viewpoints, working distance: {wd_m * 1000:.1f} mm")

    print("[2/5] Building camera poses + augmentation variants...")
    world_poses = build_camera_poses(positions, normals, wd_m)
    lock_nominal_wrist3 = not (args.roll_augment or args.tilt_augment)
    settings = build_settings(
        object_position=config.TARGET_OBJECT["position"],
        object_quat_wxyz=config.TARGET_OBJECT["rotation"],
        working_distance_m=wd_m,
        roll_augment=args.roll_augment, roll_step_deg=args.roll_step_deg,
        tilt_augment=args.tilt_augment, tilt_angles_deg=args.tilt_angles_deg,
        tilt_azimuths=args.tilt_azimuths, dedup=args.dedup, dedup_rad=args.dedup_rad,
        num_seeds=args.num_seeds, ik_seed=args.ik_seed,
        lock_nominal_wrist3=lock_nominal_wrist3,
    )
    targets = _build_pose_variants(
        world_poses, wd_m, roll_augment=args.roll_augment, roll_step_deg=args.roll_step_deg,
        tilt_augment=args.tilt_augment, tilt_angles_deg=args.tilt_angles_deg,
        tilt_azimuths=args.tilt_azimuths,
    )
    print(f"  {len(targets['position'])} pose targets "
          f"({len(targets['position']) / max(n_viewpoints, 1):.0f}/viewpoint); "
          f"roll={args.roll_augment}, tilt={args.tilt_augment}, dedup={args.dedup}")

    print("[3/5] Multi-seed IK + dedup...")
    world_config = build_collision_world(args.object)
    robot_cfg = resolve_robot_config(ROBOT_CONFIG)
    print(f"  Robot YAML: urdf={robot_cfg['robot_cfg']['kinematics']['urdf_path']}")
    joint_lower, joint_upper, joint_periods = _joint_limits_and_periods(robot_cfg)
    reference = _parse_joints(args.reference_joints)
    if reference is not None:
        print(f"  Aligning IK to reference joints "
              f"{np.round(np.rad2deg(reference), 1).tolist()} deg")
    wrist3_fixed = float(config.ROBOT_START_STATE[-1])
    representatives, metadata = _solve_pose_variant_candidates(
        targets, n_viewpoints, world_config, robot_cfg, args.num_seeds, args.batch_size,
        wrist3_fixed, lock_nominal_wrist3=lock_nominal_wrist3, joint_periods=joint_periods,
        ik_seed=args.ik_seed, dedup_rad=settings["dedup_rad"],
        reference_joints=reference, joint_lower=joint_lower, joint_upper=joint_upper,
    )

    print("[4/5] Collision filter...")
    removed = _collision_filter_representatives(representatives, robot_cfg, world_config, metadata)
    success_counts = np.asarray([len(r) for r in representatives], dtype=np.int32)
    reachable = success_counts > 0
    reachable_count = int(reachable.sum())
    total = int(success_counts.sum())
    print(f"  {total} candidates, collision-removed: {removed}, "
          f"reachable: {reachable_count}/{n_viewpoints}")

    print("[5/5] Writing result + saving IK...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "object": args.object,
        "viewpoints": str(h5_path),
        "num_viewpoints": int(n_viewpoints),
        "working_distance_m": float(wd_m),
        "num_seeds": int(args.num_seeds),
        "roll_augment": bool(args.roll_augment),
        "tilt_augment": bool(args.tilt_augment),
        "dedup": bool(args.dedup),
        "object_position": config.TARGET_OBJECT["position"].astype(float).tolist(),
        "object_quat_wxyz": config.TARGET_OBJECT["rotation"].astype(float).tolist(),
        "success_counts": success_counts.astype(int).tolist(),
        "reachable": reachable.astype(bool).tolist(),
        "reachable_count": reachable_count,
        "total_ik_success": total,
        "created_at": time.time(),
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"  Reachable viewpoints: {reachable_count}/{n_viewpoints} "
        f"({100.0 * reachable_count / max(n_viewpoints, 1):.1f}%), "
        f"candidates: {total}"
    )
    print(f"IK_REACHABILITY_JSON {output_path}")

    save_path = (
        Path(args.save_ik) if args.save_ik
        else ik_solutions_path(h5_path, roll_augment=args.roll_augment,
                               tilt_augment=args.tilt_augment, dedup=args.dedup)
    )
    n_saved = save_ik_solutions(
        save_path, representatives, metadata, settings,
        source_viewpoints=h5_path, object_name=args.object,
    )
    print(f"  Saved {n_saved} IK candidates across {reachable_count} reachable viewpoints")
    print(f"IK_SOLUTIONS_H5 {save_path}")


if __name__ == "__main__":
    main()

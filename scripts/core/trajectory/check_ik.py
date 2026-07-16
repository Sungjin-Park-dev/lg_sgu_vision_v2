#!/usr/bin/env python3
"""
Check per-viewpoint IK reachability for the current object pose.

This is the lightweight counterpart to trajectory/cli.py for Isaac UI
placement feedback. It runs only the Phase 1 multi-seed IK stage and writes a
JSON file containing one success count per viewpoint, preserving the HDF5 order
so the UI can color the displayed points directly.
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
    IK_BATCH_SIZE,
    NUM_IK_SEEDS,
    ROBOT_CONFIG,
    resolve_robot_config,
    build_camera_poses,
    build_collision_world,
    rot_to_quat_batch,
    solve_ik_multi_seed,
)
from core.viewpoint import load_viewpoints_hdf5  # noqa: E402
from common import config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check which viewpoints have at least one cuRobo IK solution",
    )
    parser.add_argument("--object", type=str, required=True, help="Object name")
    parser.add_argument("--num-viewpoints", type=int, default=None,
                        help="Number of viewpoints, only used if --viewpoints is omitted")
    parser.add_argument("--viewpoints", type=str, default=None,
                        help="Direct path to viewpoints.h5")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--num-seeds", type=int, default=NUM_IK_SEEDS,
                        help=f"IK seeds per viewpoint (default: {NUM_IK_SEEDS})")
    parser.add_argument("--batch-size", type=int, default=IK_BATCH_SIZE,
                        help=f"IK batch size (default: {IK_BATCH_SIZE})")
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
    if args.viewpoints is None and args.num_viewpoints is None:
        parser.error("Either --viewpoints or --num-viewpoints is required")
    return args


def main() -> None:
    args = _parse_args()

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
        else config.get_viewpoint_path(args.object, args.num_viewpoints)
    )

    print("[1/4] Loading viewpoints...")
    viewpoint = load_viewpoints_hdf5(h5_path)
    positions = viewpoint.positions
    normals = viewpoint.normals
    path_order = viewpoint.path_order
    wd_m = viewpoint.working_distance_m
    print(f"  Loaded from {h5_path}")
    print(f"  {len(positions)} viewpoints, working distance: {wd_m * 1000:.1f} mm")
    if path_order is not None:
        print("  Preserving raw h5 order for UI point-color alignment")

    print("[2/4] Building camera poses...")
    world_poses = build_camera_poses(positions, normals, wd_m)
    positions_np = world_poses[:, :3, 3]
    quats_np = rot_to_quat_batch(world_poses[:, :3, :3])
    print(f"  {len(world_poses)} camera poses built")

    print("[3/4] Running multi-seed IK...")
    world_config = build_collision_world(args.object)
    robot_cfg = resolve_robot_config(ROBOT_CONFIG)
    print(f"  Robot YAML: urdf={robot_cfg['robot_cfg']['kinematics']['urdf_path']}")

    _, all_success = solve_ik_multi_seed(
        robot_cfg,
        world_config,
        positions_np,
        quats_np,
        num_seeds=args.num_seeds,
        batch_size=args.batch_size,
    )

    success_counts = all_success.sum(axis=1).astype(np.int32)
    reachable = success_counts > 0
    reachable_count = int(reachable.sum())
    total_success = int(success_counts.sum())

    print("[4/4] Writing result...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "object": args.object,
        "viewpoints": str(h5_path),
        "num_viewpoints": int(len(success_counts)),
        "working_distance_m": float(wd_m),
        "num_seeds": int(args.num_seeds),
        "batch_size": int(args.batch_size),
        "object_position": config.TARGET_OBJECT["position"].astype(float).tolist(),
        "object_quat_wxyz": config.TARGET_OBJECT["rotation"].astype(float).tolist(),
        "success_counts": success_counts.astype(int).tolist(),
        "reachable": reachable.astype(bool).tolist(),
        "reachable_count": reachable_count,
        "total_ik_success": total_success,
        "created_at": time.time(),
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"  Reachable viewpoints: {reachable_count}/{len(success_counts)} "
        f"({100.0 * reachable_count / max(len(success_counts), 1):.1f}%), "
        f"IK solutions: {total_success}/{len(success_counts) * args.num_seeds}"
    )
    print(f"IK_REACHABILITY_JSON {output_path}")


if __name__ == "__main__":
    main()

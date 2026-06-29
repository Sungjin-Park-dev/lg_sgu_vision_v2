#!/usr/bin/env python3
"""Solve Delaunay-constrained viewpoint/IK GTSP components with GLNS.jl.

This is an experimental, standalone stage. It computes fresh collision-aware
IK candidates, solves one open GTSP per induced Delaunay component, and writes a
``glns_result_*.h5`` artifact for ``scripts/apps/trajectory_studio.py``. It does
not alter the source viewpoint HDF5 or invoke the trajectory/motion planner.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config  # noqa: E402
from common.glns_utils import (  # noqa: E402
    build_gtsp_problem,
    decode_and_validate_tour,
    expand_edges_by_hops,
    find_hamiltonian_open_path,
    induce_adjacency,
    parse_glns_tour,
    write_result_hdf5,
    write_simple_gtsp,
)
from core import plan_trajectory as PT  # noqa: E402


JULIA_PROJECT = PROJECT_ROOT / "scripts" / "julia" / "glns"
JULIA_WRAPPER = JULIA_PROJECT / "run_glns.jl"
DEFAULT_FEASIBILITY_TIMEOUT_S = 5.0
DEFAULT_MAX_MATRIX_MIB = 512.0

# Joint-differentiated reconfiguration cost (default). base = pan/lift/elbow must
# stay put within a component; wrist (1/2/3) may reconfigure cheaply. L2 tiebreak
# weights gradient base>elbow>wrist>roll so even sub-threshold drift favours the
# base staying still. See docs/glns_path.md.
DEFAULT_JOINT_WEIGHTS = (1.0, 1.0, 0.5, 0.2, 0.2, 0.05)
RECONFIG_BASE_JOINTS = (0, 1, 2)
RECONFIG_WRIST_JOINTS = (3, 4, 5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delaunay-only component GTSP over viewpoints and IK candidates",
    )
    parser.add_argument("--object", required=True, help="Object name")
    parser.add_argument("--num-viewpoints", type=int, default=None,
                        help="Viewpoint count, used when --viewpoints is omitted")
    parser.add_argument("--viewpoints", type=Path, default=None,
                        help="Source viewpoints HDF5 containing viewpoints/adjacency")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output GLNS result HDF5 (default: timestamped data/{object}/ik/...)")
    parser.add_argument("--num-seeds", type=int, default=PT.NUM_IK_SEEDS,
                        help=f"IK seeds per viewpoint (default: {PT.NUM_IK_SEEDS})")
    parser.add_argument("--ik-batch-size", type=int, default=PT.IK_BATCH_SIZE,
                        help=f"IK GPU batch size (default: {PT.IK_BATCH_SIZE})")
    parser.add_argument("--reconfig-threshold-deg", type=float,
                        default=PT.RECONFIG_THRESHOLD_DEG,
                        help=f"L-inf reconfiguration threshold (default: {PT.RECONFIG_THRESHOLD_DEG})")
    parser.add_argument("--delaunay-expand-hops", type=int, default=1,
                        help="그래프 완화: Delaunay 를 N-hop 까지 이웃으로 확장(1=순수 Delaunay, "
                             "2=이웃의 이웃까지 허용 → GLNS 순서 자유도↑로 reconfig 회피 여지). default 1")
    parser.add_argument("--joint-weights", type=float, nargs=6, default=None,
                        metavar=("PAN", "LIFT", "ELBOW", "W1", "W2", "W3"),
                        help="per-joint L2 동점-깨기 가중치 "
                             f"(default 차등 {list(DEFAULT_JOINT_WEIGHTS)})")
    parser.add_argument("--reconfig-weight-base", type=float, default=12.0,
                        help="base(pan/lift/elbow) reconfig 1회 비용 = N×wrist 1회 (default 12)")
    parser.add_argument("--reconfig-weight-wrist", type=float, default=1.0,
                        help="wrist reconfig 1회 비용 단위 (default 1)")
    parser.add_argument("--uniform-reconfig", action="store_true",
                        help="옛 동작: 6-DoF 단일 binary reconfig + 균일 L2 가중치(비교용). "
                             "base/wrist 차등을 끈다.")
    parser.add_argument("--roll-augment", action="store_true",
                        help="광축 roll 둘레 IK 후보 증강 + 5-DoF(wr3 제외) reconfig 비용 "
                             "(wrist flip 대신 rolled 자세로 pointing 연속 해 선택)")
    parser.add_argument("--roll-step-deg", type=float, default=30.0,
                        help="[--roll-augment] roll sweep 간격 deg (default: 30 → 12 각도)")
    parser.add_argument("--glns-mode", choices=("fast", "default", "slow"), default="fast")
    parser.add_argument("--glns-timeout", type=int, default=30,
                        help="GLNS max time per component in seconds (default: 30)")
    parser.add_argument("--glns-seed", type=int, default=42)
    parser.add_argument("--julia", default="julia", help="Julia executable")
    parser.add_argument("--julia-project", type=Path, default=JULIA_PROJECT)
    parser.add_argument("--feasibility-timeout", type=float,
                        default=DEFAULT_FEASIBILITY_TIMEOUT_S)
    parser.add_argument("--max-matrix-mib", type=float, default=DEFAULT_MAX_MATRIX_MIB,
                        help="Refuse a component whose dense Int64 matrix exceeds this size")
    parser.add_argument("--keep-glns-files", action="store_true",
                        help="Keep generated .gtsp and GLNS tour files beside the result")
    parser.add_argument("--object-position", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--object-quat", type=float, nargs=4, default=None,
                        metavar=("W", "X", "Y", "Z"))
    args = parser.parse_args()

    if args.viewpoints is None and args.num_viewpoints is None:
        parser.error("Either --viewpoints or --num-viewpoints is required")
    if args.num_seeds <= 0 or args.ik_batch_size <= 0:
        parser.error("--num-seeds and --ik-batch-size must be > 0")
    if args.reconfig_threshold_deg <= 0.0:
        parser.error("--reconfig-threshold-deg must be > 0")
    if not 0.0 < args.roll_step_deg <= 180.0:
        parser.error("--roll-step-deg must be in (0, 180]")
    if args.glns_timeout <= 0 or args.feasibility_timeout <= 0.0:
        parser.error("solver timeouts must be > 0")
    if args.max_matrix_mib <= 0.0:
        parser.error("--max-matrix-mib must be > 0")
    return args


def _load_source(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Viewpoint HDF5 not found: {path}")
    with h5py.File(path, "r") as f:
        group = f["viewpoints"]
        positions = np.asarray(group["positions"], dtype=np.float64)
        normals = np.asarray(group["normals"], dtype=np.float64)
        if "adjacency" not in group or "edges" not in group["adjacency"]:
            raise ValueError(
                f"{path} has no viewpoints/adjacency/edges. Regenerate viewpoints with "
                "the current generate_viewpoints.py (without --no-delaunay)."
            )
        adjacency = group["adjacency"]
        edges = np.asarray(adjacency["edges"], dtype=np.int32)
        wd_m = config.CAMERA_WORKING_DISTANCE_MM / 1000.0
        if "metadata" in f and "camera_spec" in f["metadata"]:
            wd_m = float(
                f["metadata/camera_spec"].attrs.get(
                    "working_distance_mm", config.CAMERA_WORKING_DISTANCE_MM,
                )
            ) / 1000.0
    return {"positions": positions, "normals": normals, "edges": edges, "wd_m": wd_m}


def _check_glns_environment(julia: str, project: Path) -> str:
    executable = shutil.which(julia) if os.path.sep not in julia else julia
    if executable is None or not Path(executable).exists():
        raise RuntimeError(f"Julia executable not found: {julia}")
    command = [
        str(executable), f"--project={project}", "--startup-file=no",
        "-e", "using GLNS; print(\"GLNS_OK\")",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0 or "GLNS_OK" not in result.stdout:
        setup = f"{executable} --project={project} -e 'using Pkg; Pkg.instantiate()'"
        detail = (result.stderr or result.stdout).strip().splitlines()
        tail = detail[-1] if detail else "unknown Julia error"
        raise RuntimeError(f"GLNS.jl environment is not ready ({tail}). Run:\n  {setup}")
    return str(executable)


def _run_glns(
    executable: str,
    project: Path,
    instance: Path,
    tour_path: Path,
    mode: str,
    timeout_s: int,
    seed: int,
) -> float:
    started = time.perf_counter()
    command = [
        executable, f"--project={project}", "--startup-file=no", str(JULIA_WRAPPER),
        str(instance), str(tour_path), mode, str(timeout_s), str(seed),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True,
        timeout=timeout_s + 60, check=False,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"GLNS process failed ({result.returncode}): {detail[-2000:]}")
    if not tour_path.exists():
        raise RuntimeError("GLNS completed without writing its tour file")
    return elapsed


def _collision_filter_representatives(representatives, robot_cfg, world):
    offsets = []
    flat = []
    cursor = 0
    for viewpoint, reps in enumerate(representatives):
        if len(reps):
            offsets.append((viewpoint, cursor, len(reps)))
            flat.append(reps)
            cursor += len(reps)
    if not flat:
        return 0
    colliding, _ = PT.batch_collision_check(np.concatenate(flat, axis=0), robot_cfg, world)
    removed = 0
    for viewpoint, start, count in offsets:
        free = ~colliding[start:start + count]
        removed += count - int(free.sum())
        representatives[viewpoint] = representatives[viewpoint][free]
    return removed


def _default_output(object_name: str, count: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.get_ik_path(object_name, count, f"glns_result_{stamp}.h5")


def _roll_augmented_representatives(world_poses, world, robot_cfg,
                                   num_seeds, batch_size, roll_step_deg):
    """viewpoint마다 광축(+z) 둘레 roll 각도들로 IK 후보 생성 (wr3 고정 안 함).

    rolled pose 를 IK 로 풀면 pan~wrist_2 분기가 다양하게 확보돼, GLNS 가 wrist flip 대신
    rolled 자세로 이웃과 pointing 연속인 해를 고를 수 있다. wr3 는 각 해가 실제로 가진 값을
    유지한다(나중에 5-DoF reconfig 비용이 wr3 차이를 reconfig 로 안 세고, 2차 L2 가 과한
    roll 만 억제). solve_ik_multi_seed 가 내부에서 normalize_joints 처리.
    """
    angles = np.arange(0.0, 360.0, roll_step_deg)
    n_roll = len(angles)
    n_vp = len(world_poses)
    rotz = np.stack([Rotation.from_euler("z", a, degrees=True).as_matrix() for a in angles])
    Rs = np.empty((n_vp * n_roll, 3, 3), dtype=np.float64)
    ps = np.empty((n_vp * n_roll, 3), dtype=np.float64)
    for v in range(n_vp):
        Rs[v * n_roll:(v + 1) * n_roll] = world_poses[v, :3, :3] @ rotz   # (n_roll,3,3)
        ps[v * n_roll:(v + 1) * n_roll] = world_poses[v, :3, 3]
    quats = PT.rot_to_quat_batch(Rs)
    print(f"  Roll-augmented IK: {n_vp} viewpoints × {n_roll} rolls "
          f"(step {roll_step_deg:.0f}°) = {n_vp * n_roll} poses...")
    sols, succ = PT.solve_ik_multi_seed(
        robot_cfg, world, ps, quats, num_seeds=num_seeds, batch_size=batch_size,
    )
    seeds, dof = sols.shape[1], sols.shape[2]
    sols = sols.reshape(n_vp, n_roll * seeds, dof)   # viewpoint 별로 전 roll 후보 묶기
    succ = succ.reshape(n_vp, n_roll * seeds)
    return PT.cluster_ik_solutions(sols, succ)


def main() -> int:
    args = _parse_args()
    if config.apply_object_placement(args.object):
        print(f"  Per-object placement: pos={config.TARGET_OBJECT['position']}, "
              f"quat={config.TARGET_OBJECT['rotation']}")
    if args.object_position is not None:
        config.TARGET_OBJECT["position"] = np.asarray(args.object_position, dtype=np.float64)
    if args.object_quat is not None:
        config.TARGET_OBJECT["rotation"] = np.asarray(args.object_quat, dtype=np.float64)

    # Resolve the joint-differentiated reconfiguration cost. --uniform-reconfig
    # collapses both tiers into a single 6-DoF binary with even L2 weights (the
    # legacy behaviour) for A/B comparison.
    if args.uniform_reconfig:
        joint_weights = np.ones(6, dtype=np.float64)
        base_joints, wrist_joints = (0, 1, 2, 3, 4, 5), ()
        weight_base, weight_wrist = 1.0, 1.0   # 옛 단일-binary magnitude 재현
    else:
        joint_weights = np.asarray(
            args.joint_weights if args.joint_weights is not None else DEFAULT_JOINT_WEIGHTS,
            dtype=np.float64,
        )
        base_joints, wrist_joints = RECONFIG_BASE_JOINTS, RECONFIG_WRIST_JOINTS
        weight_base, weight_wrist = args.reconfig_weight_base, args.reconfig_weight_wrist
    print(f"  Reconfig cost: base joints={base_joints}, wrist joints={wrist_joints}, "
          f"L2 weights={joint_weights.round(3).tolist()}, "
          f"penalty base:wrist = {weight_base}:{weight_wrist}")
    # Effective per-tier joint indices used to *re-derive* is_reconfiguration from
    # the chosen tour; must mirror build_gtsp_problem (roll-augment drops wr3).
    base_idx_arr = np.asarray(sorted(set(base_joints)), dtype=int)
    wrist_idx_arr = np.asarray(
        sorted(set(wrist_joints) - ({5} if args.roll_augment else set())), dtype=int,
    )

    source_path = (args.viewpoints.resolve() if args.viewpoints is not None
                   else config.get_viewpoint_path(args.object, args.num_viewpoints).resolve())
    print("[1/6] Validating Julia/GLNS environment...")
    julia = _check_glns_environment(args.julia, args.julia_project.resolve())

    print("[2/6] Loading raw-index viewpoints and Delaunay graph...")
    source = _load_source(source_path)
    positions, normals = source["positions"], source["normals"]
    n_viewpoints = len(positions)
    source_edges = source["edges"]
    print(f"  {n_viewpoints} viewpoints, {len(source_edges)} Delaunay edges")
    if args.delaunay_expand_hops > 1:
        graph_edges = expand_edges_by_hops(source_edges, n_viewpoints, args.delaunay_expand_hops)
        print(f"  Graph relaxed to {args.delaunay_expand_hops}-hop: "
              f"{len(source_edges)} → {len(graph_edges)} edges "
              f"(GLNS 순서 자유도↑)")
    else:
        graph_edges = source_edges

    print("[3/6] Computing fresh collision-aware IK candidates...")
    world_poses = PT.build_camera_poses(positions, normals, source["wd_m"])
    pose_positions = world_poses[:, :3, 3]
    pose_quats = PT.rot_to_quat_batch(world_poses[:, :3, :3])
    world = PT.build_collision_world(args.object)
    robot_cfg = PT._resolve_robot_config(PT.ROBOT_CONFIG)
    wrist3_fixed = float(config.ROBOT_START_STATE[-1])
    if args.roll_augment:
        representatives = _roll_augmented_representatives(
            world_poses, world, robot_cfg, args.num_seeds, args.ik_batch_size,
            args.roll_step_deg,
        )
        # wr3 고정 안 함 — rolled 후보의 실제 wr3 유지(5-DoF reconfig 비용이 처리).
    else:
        all_solutions, all_success = PT.solve_ik_multi_seed(
            robot_cfg, world, pose_positions, pose_quats,
            num_seeds=args.num_seeds, batch_size=args.ik_batch_size,
        )
        representatives = PT.cluster_ik_solutions(all_solutions, all_success)
        for reps in representatives:
            if len(reps):
                reps[:, -1] = wrist3_fixed
    removed_collision = _collision_filter_representatives(representatives, robot_cfg, world)
    candidate_counts = np.asarray([len(reps) for reps in representatives], dtype=np.int32)
    reachable = candidate_counts > 0
    print(f"  Reachable: {int(reachable.sum())}/{n_viewpoints}; "
          f"collision candidates removed: {removed_collision}")
    if not np.any(reachable):
        raise RuntimeError("No reachable viewpoints remain after IK/collision filtering")

    print("[4/6] Recomputing induced Delaunay components...")
    induced_edges, component_id, components = induce_adjacency(graph_edges, reachable)
    print(f"  {len(components)} components after dropping unreachable viewpoints")

    print("[5/6] Solving one open GTSP per component...")
    threshold_rad = np.deg2rad(args.reconfig_threshold_deg)
    component_results: list[dict] = []
    debug_root = None
    output_path = (args.output.resolve() if args.output is not None
                   else _default_output(args.object, n_viewpoints).resolve())
    if args.keep_glns_files:
        debug_root = output_path.parent / f"{output_path.stem}_glns_files"
        debug_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="delaunay_glns_") as temp_dir:
        temp_root = Path(temp_dir)
        for cid, members in enumerate(components):
            print(f"  component {cid}: {len(members)} viewpoints")
            feasibility_status, witness = find_hamiltonian_open_path(
                members, induced_edges, timeout_s=args.feasibility_timeout,
            )
            base = {
                "members": members,
                "status": "pending",
                "reason": "",
                "feasibility_witness": witness,
            }
            if witness is None:
                base.update(
                    status="infeasible",
                    reason=f"Delaunay-only Hamiltonian path precheck: {feasibility_status}",
                )
                component_results.append(base)
                print(f"    INFEASIBLE ({feasibility_status})")
                continue

            if len(members) == 1:
                viewpoint = int(members[0])
                reps = representatives[viewpoint]
                candidate = int(np.argmin(np.linalg.norm(reps - config.ROBOT_START_STATE, axis=1)))
                base.update(
                    status="solved", reason="trivial singleton", solver_cost=0,
                    reconfig_unit=1, reconfig_unit_base=1, reconfig_unit_wrist=1,
                    forbidden_cost=1, joint_cost_scale=1000,
                    viewpoint_order=np.array([viewpoint], dtype=np.int32),
                    selected_candidate_index=np.array([candidate], dtype=np.int32),
                    selected_joints=reps[candidate:candidate + 1],
                    edge_linf_rad=np.empty((0,)), edge_linf_base_rad=np.empty((0,)),
                    edge_linf_wrist_rad=np.empty((0,)), edge_l2_rad=np.empty((0,)),
                    is_reconfiguration=np.empty((0,), dtype=bool),
                    is_reconfiguration_base=np.empty((0,), dtype=bool),
                    is_reconfiguration_wrist=np.empty((0,), dtype=bool),
                    num_reconfigurations=0, num_reconfigurations_base=0,
                    num_reconfigurations_wrist=0, solver_seconds=0.0, matrix_mib=0.0,
                )
                component_results.append(base)
                continue

            n_vertices = int(candidate_counts[members].sum()) + 1
            matrix_mib = n_vertices * n_vertices * 8 / (1024 ** 2)
            if matrix_mib > args.max_matrix_mib:
                base.update(
                    status="matrix_too_large",
                    reason=f"estimated matrix {matrix_mib:.1f} MiB > limit {args.max_matrix_mib:.1f}",
                    matrix_mib=matrix_mib,
                )
                component_results.append(base)
                print(f"    SKIP ({base['reason']})")
                continue

            try:
                problem = build_gtsp_problem(
                    members, representatives, induced_edges, threshold_rad,
                    joint_weights=joint_weights,
                    reconfig_base_joints=base_joints,
                    reconfig_wrist_joints=wrist_joints,
                    reconfig_weight_base=weight_base,
                    reconfig_weight_wrist=weight_wrist,
                    reconfig_exclude_last=args.roll_augment,
                )
                instance = temp_root / f"component_{cid:03d}.gtsp"
                tour_file = temp_root / f"component_{cid:03d}.tour.txt"
                write_simple_gtsp(instance, problem)
                elapsed = _run_glns(
                    julia, args.julia_project.resolve(), instance, tour_file,
                    args.glns_mode, args.glns_timeout, args.glns_seed + cid,
                )
                decoded = decode_and_validate_tour(parse_glns_tour(tour_file), problem)
                order = decoded["viewpoint_order"]
                candidates = decoded["candidate_order"]
                selected = np.stack([
                    representatives[int(vp)][int(candidate)]
                    for vp, candidate in zip(order, candidates)
                ])
                diff = np.diff(selected, axis=0)
                absd = np.abs(diff)
                # base/wrist 그룹 L∞ 를 같은 29° 임계로 판정 — build_gtsp_problem 과 동일.
                linf_base = (np.max(absd[:, base_idx_arr], axis=1) if base_idx_arr.size
                             else np.zeros(len(diff)))
                linf_wrist = (np.max(absd[:, wrist_idx_arr], axis=1) if wrist_idx_arr.size
                              else np.zeros(len(diff)))
                linf = np.max(absd, axis=1)            # 6-DoF 전체(시각화/호환용)
                l2 = np.linalg.norm(diff, axis=1)
                is_reconfig_base = linf_base > threshold_rad
                is_reconfig_wrist = linf_wrist > threshold_rad
                is_reconfig = is_reconfig_base | is_reconfig_wrist
                base.update(
                    status="solved", solver_cost=decoded["cost"],
                    reconfig_unit=problem["reconfig_unit"],
                    reconfig_unit_base=problem["reconfig_unit_base"],
                    reconfig_unit_wrist=problem["reconfig_unit_wrist"],
                    forbidden_cost=problem["forbidden_cost"],
                    joint_cost_scale=problem["joint_cost_scale"],
                    viewpoint_order=order, selected_candidate_index=candidates,
                    selected_joints=selected, edge_linf_rad=linf,
                    edge_linf_base_rad=linf_base, edge_linf_wrist_rad=linf_wrist,
                    edge_l2_rad=l2,
                    is_reconfiguration=is_reconfig,
                    is_reconfiguration_base=is_reconfig_base,
                    is_reconfiguration_wrist=is_reconfig_wrist,
                    num_reconfigurations=int(is_reconfig.sum()),
                    num_reconfigurations_base=int(is_reconfig_base.sum()),
                    num_reconfigurations_wrist=int(is_reconfig_wrist.sum()),
                    solver_seconds=elapsed, matrix_mib=matrix_mib,
                )
                if debug_root is not None:
                    shutil.copy2(instance, debug_root / instance.name)
                    shutil.copy2(tour_file, debug_root / tour_file.name)
                print(f"    SOLVED: reconfigs={int(is_reconfig.sum())} "
                      f"(base={int(is_reconfig_base.sum())}, "
                      f"wrist={int(is_reconfig_wrist.sum())}), "
                      f"cost={decoded['cost']}, {elapsed:.2f}s")
            except Exception as exc:  # preserve other components and diagnostics
                base.update(status="solver_failed", reason=str(exc), matrix_mib=matrix_mib)
                print(f"    FAILED: {exc}")
            component_results.append(base)

    print("[6/6] Writing standalone GLNS result...")
    solved = [c for c in component_results if c["status"] == "solved"]
    failed = [c for c in component_results if c["status"] != "solved"]
    total_reconfigs = sum(int(c.get("num_reconfigurations", 0)) for c in solved)
    total_reconfigs_base = sum(int(c.get("num_reconfigurations_base", 0)) for c in solved)
    total_reconfigs_wrist = sum(int(c.get("num_reconfigurations_wrist", 0)) for c in solved)
    try:
        source_ref = str(source_path.relative_to(PROJECT_ROOT))
    except ValueError:
        source_ref = str(source_path)
    metadata = {
        "object": args.object,
        "source_viewpoints": source_ref,
        "source_viewpoint_count": n_viewpoints,
        "working_distance_m": float(source["wd_m"]),
        "object_position": config.TARGET_OBJECT["position"].astype(float).tolist(),
        "object_quat_wxyz": config.TARGET_OBJECT["rotation"].astype(float).tolist(),
        "robot_config": PT.ROBOT_CONFIG,
        "num_ik_seeds": args.num_seeds,
        "ik_batch_size": args.ik_batch_size,
        "wrist3_fixed_rad": (float("nan") if args.roll_augment else wrist3_fixed),
        "roll_augmented": bool(args.roll_augment),
        "roll_step_deg": float(args.roll_step_deg),
        "reconfig_threshold_deg": args.reconfig_threshold_deg,
        "joint_weights": joint_weights.astype(float).tolist(),
        "reconfig_base_joints": list(base_joints),
        "reconfig_wrist_joints": list(wrist_joints),
        "reconfig_weight_base": float(weight_base),
        "reconfig_weight_wrist": float(weight_wrist),
        "uniform_reconfig": bool(args.uniform_reconfig),
        "delaunay_expand_hops": int(args.delaunay_expand_hops),
        "graph_edge_count": int(len(graph_edges)),
        "glns_mode": args.glns_mode,
        "glns_timeout_s": args.glns_timeout,
        "glns_seed": args.glns_seed,
        "reachable_count": int(reachable.sum()),
        "dropped_unreachable": int((~reachable).sum()),
        "num_components": len(components),
        "solved_components": len(solved),
        "failed_components": len(failed),
        "total_reconfigurations": total_reconfigs,
        "total_reconfigurations_base": total_reconfigs_base,
        "total_reconfigurations_wrist": total_reconfigs_wrist,
        "created_at": datetime.now().isoformat(),
    }
    write_result_hdf5(
        output_path, metadata, reachable, candidate_counts,
        induced_edges, component_id, component_results,
    )
    print(f"  GLNS_RESULT_H5 {output_path}")
    print(f"  solved={len(solved)}/{len(components)}, reconfigs={total_reconfigs} "
          f"(base={total_reconfigs_base}, wrist={total_reconfigs_wrist}), "
          f"unreachable={int((~reachable).sum())}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

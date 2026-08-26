#!/usr/bin/env python3
"""Solve Delaunay-constrained viewpoint/IK GTSP components with GLNS.jl.

궤적 생성의 1단계다. 충돌-aware IK 후보를 새로 계산하고, 유도된 Delaunay 성분마다 open
GTSP 를 하나씩 풀어 ``data/{object}/trajectory/{N}/solution.h5`` 를 쓴다. 그 해를 실행 가능한
궤적으로 바꾸는 것은 2단계인 ``glns/verify.py`` 다(모션 계획 + 충돌 게이트).
원본 viewpoint HDF5 는 건드리지 않는다.
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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config, scene_config  # noqa: E402
from core import trajectory as PT  # noqa: E402
from core.glns.candidates import (  # noqa: E402
    _build_pose_variants,
    _collision_filter_representatives,
    _joint_limits_and_periods,
    _solve_pose_variant_candidates,
)
from core.glns.ik_store import (  # noqa: E402
    build_settings,
    ik_solutions_path,
    load_ik_solutions,
    save_ik_solutions,
)
from core.glns.problem import (  # noqa: E402
    build_gtsp_problem,
    effective_candidate_cap,
    expand_edges_by_hops,
    find_hamiltonian_open_path,
    induce_adjacency,
    periodic_joint_delta,
    prune_candidate_sets,
    unwrap_joint_path,
)
from core.glns.storage import (  # noqa: E402
    decode_and_validate_tour,
    parse_glns_tour,
    write_result_hdf5,
    write_simple_gtsp,
)
from core.viewpoint import load_viewpoints_hdf5  # noqa: E402


JULIA_PROJECT = PROJECT_ROOT / "scripts" / "julia" / "glns"
JULIA_WRAPPER = JULIA_PROJECT / "run_glns.jl"
DEFAULT_FEASIBILITY_TIMEOUT_S = 5.0
DEFAULT_MAX_MATRIX_MIB = 512.0
DEFAULT_MATRIX_TARGET_MIB = 256.0
DEFAULT_MAX_CANDIDATES = 16

# Joint-differentiated reconfiguration cost (default). base = pan/lift/elbow must
# stay put within a component; wrist (1/2/3) may reconfigure cheaply. L2 tiebreak
# weights gradient base>elbow>wrist>roll so even sub-threshold drift favours the
# base staying still. See docs/glns_path.md.
DEFAULT_JOINT_WEIGHTS = (1.0, 1.0, 0.5, 0.2, 0.2, 0.05)


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
                        help="GLNS 해 HDF5 (기본: data/{object}/trajectory/{N}/solution.h5)")
    parser.add_argument("--num-seeds", type=int, default=PT.NUM_IK_SEEDS,
                        help=f"IK seeds per viewpoint (default: {PT.NUM_IK_SEEDS})")
    parser.add_argument("--ik-batch-size", type=int, default=PT.IK_BATCH_SIZE,
                        help=f"IK GPU batch size (default: {PT.IK_BATCH_SIZE})")
    parser.add_argument("--ik-seed", type=int, default=PT.IK_RANDOM_SEED,
                        help=f"Deterministic cuRobo IK seed (default: {PT.IK_RANDOM_SEED})")
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
    parser.add_argument("--roll-augment", action="store_true",
                        help="add nonzero optical-axis roll IK pose variants")
    parser.add_argument("--roll-step-deg", type=float, default=30.0,
                        help="[--roll-augment] nonzero roll sweep 간격 deg (default: 30 → 11 poses)")
    parser.add_argument("--tilt-augment", action="store_true",
                        help="nominal camera XY axes around off-normal tilt IK poses")
    parser.add_argument("--tilt-angles-deg", type=float, nargs="+", default=[5.0, 10.0],
                        help="[--tilt-augment] tilt magnitudes (default: 5 10)")
    parser.add_argument("--tilt-azimuths", type=int, default=8,
                        help="[--tilt-augment] evenly spaced nominal-XY tilt axes (default: 8)")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false",
                        help="IK 후보 near-duplicate 제거를 끈다(전부 후보로 남긴다)")
    parser.set_defaults(dedup=True)
    parser.add_argument("--dedup-rad", type=float, default=PT.CANDIDATE_DEDUP_RAD,
                        help=f"[dedup] L∞ 관절 임계 rad (default: {PT.CANDIDATE_DEDUP_RAD})")
    parser.add_argument("--no-ik-reuse", action="store_true",
                        help="저장된 IK(ik_*.h5) 재사용을 끄고 항상 새로 계산한다")
    parser.add_argument("--max-candidates-per-viewpoint", type=int,
                        default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--matrix-target-mib", type=float, default=DEFAULT_MATRIX_TARGET_MIB,
                        help="automatic candidate-cap target per component (default: 256)")
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
    scene_config.add_cli_argument(parser)
    parser.add_argument("--object-position", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--object-quat", type=float, nargs=4, default=None,
                        metavar=("W", "X", "Y", "Z"))
    args = parser.parse_args()

    if args.viewpoints is None and args.num_viewpoints is None:
        parser.error("Either --viewpoints or --num-viewpoints is required")
    if args.num_seeds <= 0 or args.ik_batch_size <= 0:
        parser.error("--num-seeds and --ik-batch-size must be > 0")
    if args.ik_seed < 0:
        parser.error("--ik-seed must be >= 0")
    if args.reconfig_threshold_deg <= 0.0:
        parser.error("--reconfig-threshold-deg must be > 0")
    if not 0.0 < args.roll_step_deg <= 180.0:
        parser.error("--roll-step-deg must be in (0, 180]")
    if any(a <= 0.0 or a > 45.0 for a in args.tilt_angles_deg):
        parser.error("--tilt-angles-deg values must be in (0, 45]")
    if args.tilt_azimuths < 1 or args.max_candidates_per_viewpoint < 1:
        parser.error("--tilt-azimuths and --max-candidates-per-viewpoint must be positive")
    if args.glns_timeout <= 0 or args.feasibility_timeout <= 0.0:
        parser.error("solver timeouts must be > 0")
    if args.max_matrix_mib <= 0.0 or args.matrix_target_mib <= 0.0:
        parser.error("matrix limits must be > 0")
    if args.matrix_target_mib > args.max_matrix_mib:
        parser.error("--matrix-target-mib must not exceed --max-matrix-mib")
    return args


def _load_source(path: Path) -> dict:
    viewpoint = load_viewpoints_hdf5(path)
    if viewpoint.adjacency is None:
        raise ValueError(
            f"{path} has no viewpoints/adjacency/edges. Regenerate viewpoints with "
            "scripts/core/viewpoint/cli.py (without --no-delaunay)."
        )
    return {
        "positions": viewpoint.positions,
        "normals": viewpoint.normals,
        "edges": viewpoint.adjacency.edges,
        "wd_m": viewpoint.working_distance_m,
    }


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


def _default_output(object_name: str, count: int) -> Path:
    """물체·viewpoint 수당 해는 하나다 — 재solve 하면 덮어쓴다.

    옛 궤적 npz 는 mtime 비교로 자동 무효화되므로(trajectory_studio) 이력을 남길 필요가 없다.
    """
    return config.get_solution_path(object_name, count)


def _path_reconfig_fields(selected, threshold_rad, base_idx_arr, wrist_idx_arr) -> dict:
    """selected (M,6) → edge L∞/L2 + base/wrist reconfig 필드(solve 직후 계산과 동일 규칙)."""
    selected = np.asarray(selected, dtype=np.float64)
    diff = np.diff(selected, axis=0)
    absd = np.abs(diff)
    n = len(diff)
    linf_base = (np.max(absd[:, base_idx_arr], axis=1) if (n and base_idx_arr.size)
                 else np.zeros(n))
    linf_wrist = (np.max(absd[:, wrist_idx_arr], axis=1) if (n and wrist_idx_arr.size)
                  else np.zeros(n))
    linf = np.max(absd, axis=1) if n else np.empty((0,))
    l2 = np.linalg.norm(diff, axis=1) if n else np.empty((0,))
    is_rb = linf_base > threshold_rad
    is_rw = linf_wrist > threshold_rad
    is_r = is_rb | is_rw
    return dict(
        edge_linf_rad=linf, edge_linf_base_rad=linf_base, edge_linf_wrist_rad=linf_wrist,
        edge_l2_rad=l2, is_reconfiguration=is_r,
        is_reconfiguration_base=is_rb, is_reconfiguration_wrist=is_rw,
        num_reconfigurations=int(is_r.sum()),
        num_reconfigurations_base=int(is_rb.sum()),
        num_reconfigurations_wrist=int(is_rw.sum()),
    )


def main() -> int:
    args = _parse_args()
    # 씬을 먼저 반영한다 — 물체 배치(object_placements)가 이제 씬 소유다.
    scene_config.apply_cli(args, config)
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
    joint_weights = np.asarray(
        args.joint_weights if args.joint_weights is not None else DEFAULT_JOINT_WEIGHTS,
        dtype=np.float64,
    )
    # 계층 순위(base 개수 → 6축 개수 → tilt → 가중 L2)는 build_gtsp_problem 에 고정돼 있다.
    # 여기서 조절할 수 있는 것은 최하위 동점-깨기 가중치뿐이라, 그것만 보고한다.
    print(f"  Cost tie-break L2 weights = {joint_weights.round(3).tolist()}")
    # Strict objective/verifier definitions: q0:q3 base, q0:q6 any.
    base_idx_arr = np.arange(3, dtype=int)
    any_idx_arr = np.arange(6, dtype=int)

    source_path = (args.viewpoints.resolve() if args.viewpoints is not None
                   else config.resolve_viewpoint_path(args.object, args.num_viewpoints).resolve())
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
              f"{len(source_edges)} -> {len(graph_edges)} edges "
              f"(more ordering freedom for GLNS)")
    else:
        graph_edges = source_edges

    print(f"[3/6] Resolving collision-aware IK candidates (seed={args.ik_seed})...")
    world_poses = PT.build_camera_poses(positions, normals, source["wd_m"])
    world = PT.build_collision_world(args.object)
    robot_cfg = PT.resolve_robot_config(PT.ROBOT_CONFIG)
    joint_lower, joint_upper, joint_periods = _joint_limits_and_periods(robot_cfg)
    periodic_names = [
        name for name, period in zip(
            robot_cfg["robot_cfg"]["kinematics"]["cspace"]["joint_names"], joint_periods)
        if period > 0.0
    ]
    print(f"  Periodic joint lifting enabled: {periodic_names}")
    wrist3_fixed = float(config.ROBOT_START_STATE[-1])
    lock_nominal_wrist3 = not (args.roll_augment or args.tilt_augment)
    # Check-and-Save-IK 가 같은 물체 pose·증강·dedup 으로 저장해둔 IK 가 있으면 그대로 쓰고,
    # 아니면 새로 계산해 그 자리에 저장한다(다음 실행/모드가 재사용). ik_store 가 진실이다.
    ik_settings = build_settings(
        object_position=config.TARGET_OBJECT["position"],
        object_quat_wxyz=config.TARGET_OBJECT["rotation"],
        working_distance_m=source["wd_m"],
        roll_augment=args.roll_augment, roll_step_deg=args.roll_step_deg,
        tilt_augment=args.tilt_augment, tilt_angles_deg=args.tilt_angles_deg,
        tilt_azimuths=args.tilt_azimuths, dedup=args.dedup, dedup_rad=args.dedup_rad,
        num_seeds=args.num_seeds, ik_seed=args.ik_seed,
        lock_nominal_wrist3=lock_nominal_wrist3,
    )
    ik_path = ik_solutions_path(source_path, roll_augment=args.roll_augment,
                                tilt_augment=args.tilt_augment, dedup=args.dedup)
    loaded = None if args.no_ik_reuse else load_ik_solutions(ik_path, ik_settings)
    if loaded is not None and len(loaded[0]) != n_viewpoints:
        print(f"  Saved IK viewpoint-count mismatch - recomputing ({ik_path})")
        loaded = None

    if loaded is not None:
        representatives_raw, candidate_metadata_raw = loaded
        removed_collision = 0
        total_loaded = int(sum(len(r) for r in representatives_raw))
        print(f"  Reusing saved IK ({total_loaded} candidates, object pose + settings "
              f"match): {ik_path}")
    else:
        targets = _build_pose_variants(
            world_poses, source["wd_m"], roll_augment=args.roll_augment,
            roll_step_deg=args.roll_step_deg, tilt_augment=args.tilt_augment,
            tilt_angles_deg=args.tilt_angles_deg, tilt_azimuths=args.tilt_azimuths,
        )
        print(f"  Pose variants: {len(targets['position'])} total "
              f"({len(targets['position']) / n_viewpoints:.0f}/viewpoint)")
        representatives_raw, candidate_metadata_raw = _solve_pose_variant_candidates(
            targets, n_viewpoints, world, robot_cfg, args.num_seeds, args.ik_batch_size,
            wrist3_fixed, lock_nominal_wrist3=lock_nominal_wrist3,
            joint_periods=joint_periods, ik_seed=args.ik_seed,
            dedup_rad=ik_settings["dedup_rad"],
        )
        removed_collision = _collision_filter_representatives(
            representatives_raw, robot_cfg, world, candidate_metadata_raw,
        )
        if not args.no_ik_reuse:
            save_ik_solutions(ik_path, representatives_raw, candidate_metadata_raw,
                              ik_settings, source_viewpoints=source_path,
                              object_name=args.object)
            print(f"  Saved IK for reuse -> {ik_path}")
    candidate_counts_raw = np.asarray(
        [len(reps) for reps in representatives_raw], dtype=np.int32,
    )
    reachable = candidate_counts_raw > 0
    print(f"  Reachable: {int(reachable.sum())}/{n_viewpoints}; "
          f"collision candidates removed: {removed_collision}")
    if not np.any(reachable):
        raise RuntimeError("No reachable viewpoints remain after IK/collision filtering")

    print("[4/6] Recomputing induced Delaunay components...")
    induced_edges, component_id, components = induce_adjacency(graph_edges, reachable)
    print(f"  {len(components)} components after dropping unreachable viewpoints")

    cap_by_viewpoint = np.ones(n_viewpoints, dtype=np.int32)
    component_caps = {}
    for cid, members in enumerate(components):
        cap = effective_candidate_cap(
            len(members), args.max_candidates_per_viewpoint, args.matrix_target_mib,
        )
        cap_by_viewpoint[members] = cap
        component_caps[cid] = cap
    prune_started = time.perf_counter()
    representatives, candidate_metadata = prune_candidate_sets(
        representatives_raw, candidate_metadata_raw, induced_edges, cap_by_viewpoint,
        np.deg2rad(args.reconfig_threshold_deg), joint_weights,
        reference_joints=np.asarray(config.ROBOT_START_STATE, dtype=np.float64),
        joint_periods=joint_periods,
    )
    candidate_counts = np.asarray([len(reps) for reps in representatives], dtype=np.int32)
    prune_seconds = time.perf_counter() - prune_started
    if len(components):
        print("  Candidate caps: " + ", ".join(
            f"component {cid}=K{component_caps[cid]}" for cid in range(len(components))))
    print(f"  Candidate pruning: {int(candidate_counts_raw.sum())} raw -> "
          f"{int(candidate_counts.sum())} retained, {prune_seconds:.2f}s "
          f"({len(induced_edges)} undirected edges, no reverse recomputation)")

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
            print(f"  component {cid}: {len(members)} viewpoints, K_eff={component_caps[cid]}")
            feasibility_status, witness = find_hamiltonian_open_path(
                members, induced_edges, timeout_s=args.feasibility_timeout,
            )
            base = {
                "members": members,
                "status": "pending",
                "reason": "",
                "feasibility_witness": witness,
                "candidate_counts": candidate_counts[members],
                "candidate_counts_raw": candidate_counts_raw[members],
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
                candidate = int(np.argmin(np.linalg.norm(
                    periodic_joint_delta(reps - config.ROBOT_START_STATE, joint_periods)
                    * joint_weights, axis=1,
                )))
                selected_single = unwrap_joint_path(
                    reps[candidate:candidate + 1], joint_lower, joint_upper, joint_periods,
                    threshold_rad, joint_weights=joint_weights,
                    reference_joints=np.asarray(config.ROBOT_START_STATE, dtype=np.float64),
                )
                single_turns = np.zeros_like(selected_single, dtype=np.int16)
                periodic = joint_periods > 0.0
                single_turns[:, periodic] = np.rint(
                    (selected_single[:, periodic] - reps[candidate:candidate + 1, periodic])
                    / joint_periods[periodic]
                ).astype(np.int16)
                base.update(
                    status="solved", reason="trivial singleton", solver_cost=0,
                    reconfig_unit_base=1, base_travel_unit=1, reconfig_unit_any=1,
                    forbidden_cost=1, joint_cost_scale=1000,
                    viewpoint_order=np.array([viewpoint], dtype=np.int32),
                    selected_candidate_index=np.array([candidate], dtype=np.int32),
                    selected_joints=selected_single,
                    selected_joint_turns=single_turns,
                    edge_linf_rad=np.empty((0,)), edge_linf_base_rad=np.empty((0,)),
                    edge_linf_wrist_rad=np.empty((0,)), edge_l2_rad=np.empty((0,)),
                    is_reconfiguration=np.empty((0,), dtype=bool),
                    is_reconfiguration_base=np.empty((0,), dtype=bool),
                    is_reconfiguration_wrist=np.empty((0,), dtype=bool),
                    num_reconfigurations=0, num_reconfigurations_base=0,
                    num_reconfigurations_any=0, num_reconfigurations_wrist=0,
                    objective_base_buckets=0, objective_any_cost=0,
                    objective_tilt_cost=int(round((candidate_metadata[viewpoint]["tilt_deg"][candidate] / 5.0) ** 2)),
                    objective_joint_cost=0,
                    selected_pose_variant=np.asarray([candidate_metadata[viewpoint]["variant"][candidate]]),
                    selected_roll_deg=np.asarray([candidate_metadata[viewpoint]["roll_deg"][candidate]]),
                    selected_tilt_deg=np.asarray([candidate_metadata[viewpoint]["tilt_deg"][candidate]]),
                    selected_tilt_azimuth_deg=np.asarray([candidate_metadata[viewpoint]["tilt_azimuth_deg"][candidate]]),
                    selected_target_position=np.asarray([candidate_metadata[viewpoint]["target_position"][candidate]]),
                    selected_target_quaternion=np.asarray([candidate_metadata[viewpoint]["target_quaternion"][candidate]]),
                    solver_seconds=0.0, matrix_mib=0.0,
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
                    candidate_tilt_costs=[
                        np.rint((np.asarray(md["tilt_deg"]) / 5.0) ** 2).astype(np.int64)
                        for md in candidate_metadata
                    ],
                    joint_periods=joint_periods,
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
                selected_wrapped = np.stack([
                    representatives[int(vp)][int(candidate)]
                    for vp, candidate in zip(order, candidates)
                ])
                selected = unwrap_joint_path(
                    selected_wrapped, joint_lower, joint_upper, joint_periods,
                    threshold_rad, joint_weights=joint_weights,
                    reference_joints=np.asarray(config.ROBOT_START_STATE, dtype=np.float64),
                )
                selected_turns = np.zeros_like(selected, dtype=np.int16)
                periodic = joint_periods > 0.0
                selected_turns[:, periodic] = np.rint(
                    (selected[:, periodic] - selected_wrapped[:, periodic])
                    / joint_periods[periodic]
                ).astype(np.int16)
                diff = np.diff(selected, axis=0)
                absd = np.abs(diff)
                linf_base = np.max(absd[:, base_idx_arr], axis=1)
                linf_any = np.max(absd[:, any_idx_arr], axis=1)
                linf = np.max(absd, axis=1)            # 6-DoF 전체(시각화/호환용)
                weighted_l2 = np.linalg.norm(diff * joint_weights, axis=1)
                l2 = np.linalg.norm(diff, axis=1)
                is_reconfig_base = linf_base > threshold_rad
                is_reconfig_any = linf_any > threshold_rad
                is_reconfig = is_reconfig_any
                selected_md = [candidate_metadata[int(vp)] for vp in order]
                selected_tilt_cost = np.asarray([
                    int(round((md["tilt_deg"][int(c)] / 5.0) ** 2))
                    for md, c in zip(selected_md, candidates)
                ], dtype=np.int64)
                base.update(
                    status="solved", solver_cost=decoded["cost"],
                    reconfig_unit_base=problem["reconfig_unit_base"],
                    base_travel_unit=problem["base_travel_unit"],
                    base_bucket_rad=problem["base_bucket_rad"],
                    reconfig_unit_any=problem["reconfig_unit_any"],
                    tilt_unit=problem["tilt_unit"],
                    forbidden_cost=problem["forbidden_cost"],
                    joint_cost_scale=problem["joint_cost_scale"],
                    viewpoint_order=order, selected_candidate_index=candidates,
                    selected_joints=selected, selected_joint_turns=selected_turns,
                    edge_linf_rad=linf,
                    edge_linf_base_rad=linf_base, edge_linf_wrist_rad=linf_any,
                    edge_l2_rad=l2,
                    is_reconfiguration=is_reconfig,
                    is_reconfiguration_base=is_reconfig_base,
                    is_reconfiguration_wrist=is_reconfig_any,
                    num_reconfigurations=int(is_reconfig.sum()),
                    num_reconfigurations_base=int(is_reconfig_base.sum()),
                    num_reconfigurations_any=int(is_reconfig_any.sum()),
                    num_reconfigurations_wrist=int(is_reconfig_any.sum()),
                    objective_base_count=int(is_reconfig_base.sum()),
                    objective_base_buckets=int(np.floor(
                        linf_base / float(problem["base_bucket_rad"])).sum()),
                    objective_any_cost=int(is_reconfig_any.sum()),
                    objective_tilt_cost=int(selected_tilt_cost.sum()),
                    objective_joint_cost=int(np.rint(weighted_l2 * problem["joint_cost_scale"]).sum()),
                    selected_pose_variant=np.asarray([
                        md["variant"][int(c)] for md, c in zip(selected_md, candidates)]),
                    selected_roll_deg=np.asarray([
                        md["roll_deg"][int(c)] for md, c in zip(selected_md, candidates)]),
                    selected_tilt_deg=np.asarray([
                        md["tilt_deg"][int(c)] for md, c in zip(selected_md, candidates)]),
                    selected_tilt_azimuth_deg=np.asarray([
                        md["tilt_azimuth_deg"][int(c)] for md, c in zip(selected_md, candidates)]),
                    selected_target_position=np.stack([
                        md["target_position"][int(c)] for md, c in zip(selected_md, candidates)]),
                    selected_target_quaternion=np.stack([
                        md["target_quaternion"][int(c)] for md, c in zip(selected_md, candidates)]),
                    solver_seconds=elapsed, matrix_mib=matrix_mib,
                )
                if debug_root is not None:
                    shutil.copy2(instance, debug_root / instance.name)
                    shutil.copy2(tour_file, debug_root / tour_file.name)
                print(f"    SOLVED: reconfigs={int(is_reconfig.sum())} "
                      f"(base={int(is_reconfig_base.sum())}, "
                      f"any={int(is_reconfig_any.sum())}), "
                      f"lifted_joints={int(np.count_nonzero(selected_turns))}, "
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
    total_reconfigs_any = sum(int(c.get("num_reconfigurations_any", 0)) for c in solved)
    total_reconfigs_wrist = total_reconfigs_any  # v1 metadata compatibility
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
        # 씬을 이름이 아니라 **해결된 스냅샷**으로 박는다. 이름만 넣으면 solve 이후 YAML 이
        # 편집됐을 때 verify 가 다른 월드로 검증한다 — 실측 셀을 맞추는 동안 매일 편집한다.
        # storage.write_result_hdf5 가 dict 를 자동으로 json.dumps 하므로 저장 계층은 그대로다.
        "scene_name": config.ACTIVE_SCENE,
        "scene": config.scene_snapshot(),
        "robot_config": PT.ROBOT_CONFIG,
        "num_ik_seeds": args.num_seeds,
        "ik_batch_size": args.ik_batch_size,
        "ik_seed": args.ik_seed,
        "wrist3_fixed_rad": (float("nan") if (args.roll_augment or args.tilt_augment)
                              else wrist3_fixed),
        "roll_augmented": bool(args.roll_augment),
        "roll_step_deg": float(args.roll_step_deg),
        "tilt_augmented": bool(args.tilt_augment),
        "tilt_angles_deg": [float(x) for x in args.tilt_angles_deg],
        "tilt_azimuths": int(args.tilt_azimuths),
        "max_candidates_per_viewpoint": int(args.max_candidates_per_viewpoint),
        "matrix_target_mib": float(args.matrix_target_mib),
        "max_matrix_mib": float(args.max_matrix_mib),
        "reconfig_threshold_deg": args.reconfig_threshold_deg,
        "joint_weights": joint_weights.astype(float).tolist(),
        "joint_lower_rad": joint_lower.astype(float).tolist(),
        "joint_upper_rad": joint_upper.astype(float).tolist(),
        "joint_periods_rad": joint_periods.astype(float).tolist(),
        "joint_unwrapped": True,
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
        "total_reconfigurations_any": total_reconfigs_any,
        "total_reconfigurations_wrist": total_reconfigs_wrist,
        "created_at": datetime.now().isoformat(),
    }
    write_result_hdf5(
        output_path, metadata, reachable, candidate_counts,
        induced_edges, component_id, component_results,
        candidate_counts_raw=candidate_counts_raw,
    )
    print(f"  GLNS_RESULT_H5 {output_path}")
    print(f"  solved={len(solved)}/{len(components)}, reconfigs={total_reconfigs} "
          f"(base={total_reconfigs_base}, any={total_reconfigs_any}), "
          f"unreachable={int((~reachable).sum())}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

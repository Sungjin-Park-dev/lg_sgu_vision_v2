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
    effective_candidate_cap,
    expand_edges_by_hops,
    find_hamiltonian_open_path,
    induce_adjacency,
    parse_glns_tour,
    prune_candidate_sets,
    write_result_hdf5,
    write_simple_gtsp,
)
from core import plan_trajectory as PT  # noqa: E402


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
                        help="add nonzero optical-axis roll IK pose variants")
    parser.add_argument("--roll-step-deg", type=float, default=30.0,
                        help="[--roll-augment] nonzero roll sweep 간격 deg (default: 30 → 11 poses)")
    parser.add_argument("--tilt-augment", action="store_true",
                        help="nominal camera XY axes around off-normal tilt IK poses")
    parser.add_argument("--tilt-angles-deg", type=float, nargs="+", default=[5.0, 10.0],
                        help="[--tilt-augment] tilt magnitudes (default: 5 10)")
    parser.add_argument("--tilt-azimuths", type=int, default=8,
                        help="[--tilt-augment] evenly spaced nominal-XY tilt axes (default: 8)")
    parser.add_argument("--max-candidates-per-viewpoint", type=int,
                        default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--matrix-target-mib", type=float, default=DEFAULT_MATRIX_TARGET_MIB,
                        help="automatic candidate-cap target per component (default: 256)")
    parser.add_argument("--tilt-repair", action="store_true",
                        help="GLNS 후 강제 big-base reconfig outlier viewpoint 를 ±tilt 재-IK 로 "
                             "복구(이웃과 호환되는 해로 교체, 없으면 drop)")
    parser.add_argument("--tilt-repair-max-deg", type=float, default=10.0,
                        help="[--tilt-repair] 스캔 허용 off-normal tilt 최대각 deg (default: 10)")
    parser.add_argument("--tilt-repair-azimuths", type=int, default=8,
                        help="[--tilt-repair] tilt 방위 수 (default: 8)")
    parser.add_argument("--big-base-deg", type=float, default=120.0,
                        help="[--tilt-repair] catastrophic base reconfig 판정 L∞ deg (default: 120)")
    parser.add_argument("--outlier-max-len", type=int, default=2,
                        help="[--tilt-repair] outlier 로 볼 branch-run 최대 길이 (default: 2)")
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
    if args.tilt_repair:
        if not 0.0 < args.tilt_repair_max_deg <= 45.0:
            parser.error("--tilt-repair-max-deg must be in (0, 45]")
        if args.tilt_repair_azimuths < 1 or args.outlier_max_len < 1 or args.big_base_deg <= 0.0:
            parser.error("--tilt-repair-azimuths/--outlier-max-len/--big-base-deg must be positive")
    if args.tilt_augment and args.tilt_repair:
        parser.error("--tilt-augment and --tilt-repair are mutually exclusive")
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


def _collision_filter_representatives(representatives, robot_cfg, world, metadata=None):
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
        if metadata is not None:
            metadata[viewpoint] = {
                key: np.asarray(value)[free] for key, value in metadata[viewpoint].items()
            }
    return removed


def _default_output(object_name: str, count: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.get_ik_path(object_name, count, f"glns_result_{stamp}.h5")


def _build_pose_variants(world_poses, wd_m, *, roll_augment=False, roll_step_deg=30.0,
                         tilt_augment=False, tilt_angles_deg=(5.0, 10.0),
                         tilt_azimuths=8):
    """Return flattened nominal + roll + tilt pose targets and their metadata."""
    records = []
    roll_angles = np.arange(roll_step_deg, 360.0 - 1e-9, roll_step_deg)
    azimuths = np.linspace(0.0, 360.0, int(tilt_azimuths), endpoint=False)
    for vp, pose in enumerate(np.asarray(world_poses, dtype=np.float64)):
        R, p = pose[:3, :3], pose[:3, 3]
        surf = p + R[:, 2] * wd_m
        records.append((vp, "nominal", 0.0, 0.0, np.nan, p.copy(), R.copy()))
        if roll_augment:
            for angle in roll_angles:
                Rp = R @ Rotation.from_euler("z", angle, degrees=True).as_matrix()
                records.append((vp, "roll", float(angle), 0.0, np.nan, p.copy(), Rp))
        if tilt_augment:
            for tilt in tilt_angles_deg:
                for az in azimuths:
                    axis = R @ np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), 0.0])
                    Rp = Rotation.from_rotvec(axis * np.deg2rad(tilt)).as_matrix() @ R
                    pp = surf - Rp[:, 2] * wd_m
                    records.append((vp, "tilt", 0.0, float(tilt), float(az), pp, Rp))
    positions = np.stack([r[5] for r in records])
    rotations = np.stack([r[6] for r in records])
    return {
        "viewpoint": np.asarray([r[0] for r in records], dtype=np.int32),
        "variant": np.asarray([r[1] for r in records], dtype="U16"),
        "roll_deg": np.asarray([r[2] for r in records], dtype=np.float64),
        "tilt_deg": np.asarray([r[3] for r in records], dtype=np.float64),
        "tilt_azimuth_deg": np.asarray([r[4] for r in records], dtype=np.float64),
        "position": positions,
        "quaternion": PT.rot_to_quat_batch(rotations),
        "rotation": rotations,
    }


def _solve_pose_variant_candidates(targets, n_viewpoints, world, robot_cfg,
                                   num_seeds, batch_size, wrist3_fixed,
                                   lock_nominal_wrist3):
    sols, success = PT.solve_ik_multi_seed(
        robot_cfg, world, targets["position"], targets["quaternion"],
        num_seeds=num_seeds, batch_size=batch_size,
    )
    representatives, metadata = [], []
    for vp in range(n_viewpoints):
        kept, fields = [], {key: [] for key in (
            "variant", "roll_deg", "tilt_deg", "tilt_azimuth_deg",
            "target_position", "target_quaternion",
        )}
        for pose_idx in np.flatnonzero(targets["viewpoint"] == vp):
            for q in sols[pose_idx][success[pose_idx]]:
                q = np.asarray(q, dtype=np.float64).copy()
                if lock_nominal_wrist3:
                    q[-1] = wrist3_fixed
                if any(np.max(np.abs(q - prior)) <= PT.DP_CANDIDATE_DEDUP_RAD for prior in kept):
                    continue
                kept.append(q)
                fields["variant"].append(targets["variant"][pose_idx])
                fields["roll_deg"].append(targets["roll_deg"][pose_idx])
                fields["tilt_deg"].append(targets["tilt_deg"][pose_idx])
                fields["tilt_azimuth_deg"].append(targets["tilt_azimuth_deg"][pose_idx])
                fields["target_position"].append(targets["position"][pose_idx])
                fields["target_quaternion"].append(targets["quaternion"][pose_idx])
        representatives.append(np.asarray(kept, dtype=np.float64).reshape(-1, 6))
        metadata.append({
            key: np.asarray(value, dtype=("U16" if key == "variant" else np.float64))
            for key, value in fields.items()
        })
    return representatives, metadata


# ============================================================================
# Post-GLNS branch repair (tilt outliers that force a catastrophic base reconfig)
# ============================================================================

_BASE_IDX = np.array([0, 1, 2])


def _tilt_magnitudes(max_deg: float) -> list[float]:
    """작은 tilt 부터 시도하도록 오름차순 각도 목록(시야 손실 최소화)."""
    if max_deg <= 5.0:
        return [round(float(max_deg), 1)]
    return [round(float(max_deg) * 0.5, 1), round(float(max_deg), 1)]


def _tilt_cone_poses(world_pose, wd_m, tilt_deg, n_azimuth):
    """표면점 중심 orbit tilt 포즈들(광축 ±tilt_deg, n_azimuth 방위) → (positions, quats wxyz).

    plan_trajectory via-tilt 와 동일한 orbit: WD·표면점 유지, 광축만 기울여 새 팔 분기를 연다.
    (roll 은 wrist_3-decoupled 라 분기 불변 — tilt 만 분기를 바꾼다.)
    """
    R = np.asarray(world_pose[:3, :3], dtype=np.float64)
    p = np.asarray(world_pose[:3, 3], dtype=np.float64)
    surf = p + R[:, 2] * wd_m
    Rs, ps = [], []
    for az in np.linspace(0.0, 360.0, int(n_azimuth), endpoint=False):
        u = R @ np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), 0.0])
        u = u / np.linalg.norm(u)
        Rp = Rotation.from_rotvec(u * np.deg2rad(tilt_deg)).as_matrix() @ R
        Rs.append(Rp)
        ps.append(surf - Rp[:, 2] * wd_m)
    return np.stack(ps), PT.rot_to_quat_batch(np.stack(Rs))


def _tilt_compatible_solution(viewpoint, neighbor_configs, *, world_poses, world, robot_cfg,
                              wd_m, wrist3_fixed, num_seeds, batch_size,
                              big_base_rad, tilt_magnitudes_deg, n_azimuth):
    """outlier viewpoint 를 ±tilt 로 재-IK 해 이웃과 big-base reconfig 없는 collision-free 해 탐색.

    작은 tilt 부터 시도하고, base(어깨/팔꿈치) L∞ 가 모든 이웃에 대해 big_base_rad 미만인 해 중
    base 변화가 가장 작은 것을 고른다. 찾으면 (config (6,), used_tilt_deg), 없으면 (None, None).
    """
    for tilt_deg in tilt_magnitudes_deg:
        ps, quats = _tilt_cone_poses(world_poses[viewpoint], wd_m, tilt_deg, n_azimuth)
        sols, succ = PT.solve_ik_multi_seed(
            robot_cfg, world, ps, quats, num_seeds=num_seeds, batch_size=batch_size)
        dof = sols.shape[2]
        reps = PT.cluster_ik_solutions(sols.reshape(1, -1, dof), succ.reshape(1, -1))[0]
        if not len(reps):
            continue
        reps = np.asarray(reps, dtype=np.float64)
        reps[:, -1] = wrist3_fixed                         # wrist_3 lock(스캔 컨벤션 동일)
        colliding, _ = PT.batch_collision_check(reps, robot_cfg, world)
        reps = reps[~np.asarray(colliding, dtype=bool)]
        best, best_worst = None, np.inf
        for r in reps:
            worst = max((float(np.max(np.abs(r[_BASE_IDX] - np.asarray(nb)[_BASE_IDX])))
                         for nb in neighbor_configs), default=0.0)
            if worst < big_base_rad and worst < best_worst:
                best, best_worst = r, worst
        if best is not None:
            return best, float(tilt_deg)
    return None, None


def _repair_branch_outliers(order, selected, candidates, *, world_poses, world, robot_cfg,
                            wd_m, wrist3_fixed, num_seeds, batch_size,
                            big_base_rad, outlier_max_len, tilt_magnitudes_deg, n_azimuth):
    """GLNS 경로의 short branch-run(강제 big-base reconfig outlier)을 tilt-repair 또는 drop.

    big-base reconfig edge 가 경로를 branch-run 으로 가른다. 양쪽 run 이 모두 길면 '진짜 전환'
    으로 보고 손대지 않는다(verify 에서 via-home). 짧은 run(≤outlier_max_len, endpoint 포함)은
    minority outlier 로 보고, run 바깥 이웃 branch 와 호환되는 tilt 해를 찾으면 교체(작은 모션),
    못 찾으면 그 viewpoint 를 drop 한다. 반환 (order, selected, candidates, repaired, dropped).
    """
    order = [int(v) for v in order]
    selected = [np.asarray(q, dtype=np.float64).copy() for q in selected]
    candidates = [int(c) for c in candidates]
    M = len(order)
    repaired, dropped, drop_pos = [], [], set()
    if M < 2:
        return order, selected, candidates, repaired, dropped

    base_linf = np.array([float(np.max(np.abs(selected[i][:3] - selected[i + 1][:3])))
                          for i in range(M - 1)])
    big = base_linf > big_base_rad
    runs, s = [], 0
    for i in range(M - 1):
        if big[i]:
            runs.append((s, i))
            s = i + 1
    runs.append((s, M - 1))

    for a, b in runs:
        if (b - a + 1) > outlier_max_len:
            continue
        if a == 0 and b == M - 1:                  # big edge 없는 전체 경로 → skip
            continue
        outside = []                                # run 바깥(확정) 이웃 = 타깃 branch
        if a - 1 >= 0:
            outside.append(selected[a - 1])
        if b + 1 <= M - 1:
            outside.append(selected[b + 1])
        for p in range(a, b + 1):
            v = order[p]
            sol, tilt_deg = _tilt_compatible_solution(
                v, outside, world_poses=world_poses, world=world, robot_cfg=robot_cfg,
                wd_m=wd_m, wrist3_fixed=wrist3_fixed, num_seeds=num_seeds,
                batch_size=batch_size, big_base_rad=big_base_rad,
                tilt_magnitudes_deg=tilt_magnitudes_deg, n_azimuth=n_azimuth)
            if sol is not None:
                selected[p] = sol
                candidates[p] = -1                  # tilt-repaired(비-nominal 후보)
                repaired.append((v, tilt_deg))
            else:
                drop_pos.add(p)
                dropped.append(v)

    keep = [p for p in range(M) if p not in drop_pos]
    return ([order[p] for p in keep], [selected[p] for p in keep],
            [candidates[p] for p in keep], repaired, dropped)


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
    # Strict objective/verifier definitions: q0:q3 base, q0:q6 any.
    base_idx_arr = np.arange(3, dtype=int)
    any_idx_arr = np.arange(6, dtype=int)

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
    world = PT.build_collision_world(args.object)
    robot_cfg = PT._resolve_robot_config(PT.ROBOT_CONFIG)
    wrist3_fixed = float(config.ROBOT_START_STATE[-1])
    targets = _build_pose_variants(
        world_poses, source["wd_m"], roll_augment=args.roll_augment,
        roll_step_deg=args.roll_step_deg, tilt_augment=args.tilt_augment,
        tilt_angles_deg=args.tilt_angles_deg, tilt_azimuths=args.tilt_azimuths,
    )
    print(f"  Pose variants: {len(targets['position'])} total "
          f"({len(targets['position']) / n_viewpoints:.0f}/viewpoint)")
    representatives_raw, candidate_metadata_raw = _solve_pose_variant_candidates(
        targets, n_viewpoints, world, robot_cfg, args.num_seeds, args.ik_batch_size,
        wrist3_fixed, lock_nominal_wrist3=not (args.roll_augment or args.tilt_augment),
    )
    removed_collision = _collision_filter_representatives(
        representatives_raw, robot_cfg, world, candidate_metadata_raw,
    )
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
    representatives, candidate_metadata = prune_candidate_sets(
        representatives_raw, candidate_metadata_raw, induced_edges, cap_by_viewpoint,
        np.deg2rad(args.reconfig_threshold_deg), joint_weights,
        reference_joints=np.asarray(config.ROBOT_START_STATE, dtype=np.float64),
    )
    candidate_counts = np.asarray([len(reps) for reps in representatives], dtype=np.int32)
    if len(components):
        print("  Candidate caps: " + ", ".join(
            f"component {cid}=K{component_caps[cid]}" for cid in range(len(components))))

    print("[5/6] Solving one open GTSP per component...")
    threshold_rad = np.deg2rad(args.reconfig_threshold_deg)
    component_results: list[dict] = []
    repaired_all: list[int] = []        # tilt-repair: 교체된 outlier viewpoints
    dropped_all: list[int] = []         # tilt-repair: tilt 실패로 drop 된 outlier viewpoints
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
                    num_reconfigurations_any=0, num_reconfigurations_wrist=0,
                    objective_base_cost=0, objective_any_cost=0,
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
                    reconfig_base_joints=base_joints,
                    reconfig_wrist_joints=wrist_joints,
                    reconfig_weight_base=weight_base,
                    reconfig_weight_wrist=weight_wrist,
                    reconfig_exclude_last=False,
                    candidate_tilt_costs=[
                        np.rint((np.asarray(md["tilt_deg"]) / 5.0) ** 2).astype(np.int64)
                        for md in candidate_metadata
                    ],
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
                    reconfig_unit=problem["reconfig_unit"],
                    reconfig_unit_base=problem["reconfig_unit_base"],
                    reconfig_unit_any=problem["reconfig_unit_any"],
                    reconfig_unit_wrist=problem["reconfig_unit_any"],
                    tilt_unit=problem["tilt_unit"],
                    forbidden_cost=problem["forbidden_cost"],
                    joint_cost_scale=problem["joint_cost_scale"],
                    viewpoint_order=order, selected_candidate_index=candidates,
                    selected_joints=selected, edge_linf_rad=linf,
                    edge_linf_base_rad=linf_base, edge_linf_wrist_rad=linf_any,
                    edge_l2_rad=l2,
                    is_reconfiguration=is_reconfig,
                    is_reconfiguration_base=is_reconfig_base,
                    is_reconfiguration_wrist=is_reconfig_any,
                    num_reconfigurations=int(is_reconfig.sum()),
                    num_reconfigurations_base=int(is_reconfig_base.sum()),
                    num_reconfigurations_any=int(is_reconfig_any.sum()),
                    num_reconfigurations_wrist=int(is_reconfig_any.sum()),
                    objective_base_cost=int(is_reconfig_base.sum()),
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
                      f"cost={decoded['cost']}, {elapsed:.2f}s")
                if args.tilt_repair and len(order) >= 2:
                    try:
                        rep_order, rep_sel, rep_cand, repaired, dropped = _repair_branch_outliers(
                            order, selected, candidates,
                            world_poses=world_poses, world=world, robot_cfg=robot_cfg,
                            wd_m=source["wd_m"], wrist3_fixed=wrist3_fixed,
                            num_seeds=args.num_seeds, batch_size=args.ik_batch_size,
                            big_base_rad=np.deg2rad(args.big_base_deg),
                            outlier_max_len=args.outlier_max_len,
                            tilt_magnitudes_deg=_tilt_magnitudes(args.tilt_repair_max_deg),
                            n_azimuth=args.tilt_repair_azimuths)
                    except Exception as rexc:  # noqa: BLE001 — keep the valid un-repaired solve
                        print(f"    [tilt-repair warning] {rexc} — keeping un-repaired path")
                        repaired, dropped = [], []
                    if (repaired or dropped) and len(rep_sel) >= 1:
                        rep_sel = np.stack(rep_sel)
                        fields = _path_reconfig_fields(
                            rep_sel, threshold_rad, base_idx_arr, any_idx_arr)
                        base.update(
                            viewpoint_order=np.asarray(rep_order, dtype=np.int32),
                            selected_candidate_index=np.asarray(rep_cand, dtype=np.int32),
                            selected_joints=rep_sel, **fields)
                        repaired_all.extend(int(v) for v, _ in repaired)
                        dropped_all.extend(int(v) for v in dropped)
                        print(f"    Tilt-repair: repaired {[(v, round(d, 1)) for v, d in repaired]}, "
                              f"dropped {dropped} → base reconfigs "
                              f"{int(is_reconfig_base.sum())}→{fields['num_reconfigurations_base']}")
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
        "robot_config": PT.ROBOT_CONFIG,
        "num_ik_seeds": args.num_seeds,
        "ik_batch_size": args.ik_batch_size,
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
        "reconfig_base_joints": list(base_joints),
        "reconfig_wrist_joints": list(wrist_joints),
        "reconfig_weight_base": float(weight_base),
        "reconfig_weight_wrist": float(weight_wrist),
        "uniform_reconfig": bool(args.uniform_reconfig),
        "delaunay_expand_hops": int(args.delaunay_expand_hops),
        "graph_edge_count": int(len(graph_edges)),
        "tilt_repair": bool(args.tilt_repair),
        "tilt_repair_max_deg": float(args.tilt_repair_max_deg),
        "big_base_deg": float(args.big_base_deg),
        "outlier_max_len": int(args.outlier_max_len),
        "tilt_repaired_viewpoints": [int(v) for v in repaired_all],
        "tilt_dropped_viewpoints": [int(v) for v in dropped_all],
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
    if args.tilt_repair and (repaired_all or dropped_all):
        print(f"  tilt-repair: repaired={len(repaired_all)} {repaired_all}, "
              f"dropped={len(dropped_all)} {dropped_all}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

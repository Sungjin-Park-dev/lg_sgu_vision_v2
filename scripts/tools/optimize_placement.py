#!/usr/bin/env python3
"""물체 최적 배치 탐색 (headless) — GLNS reconfig 기준.

각 물체(data/{object})에 대해, 주어진 viewpoint + trajectory_studio 의 "Solve GLNS"
기본 파라미터 조건에서 **로봇 motion 이 가장 잘 나오는 배치(위치 XYZ)** 를 찾는다.
"잘 나온다" = GLNS solve 결과의 **reconfiguration(특히 base joint 재구성)이 최소**.

2단계 스윕:
  Stage 1 (값쌈, 배치당 수 초) — in-process cuRobo IK(`ik_backend.IKBackend`)로 XYZ 그리드
    전체의 viewpoint 도달률을 빠르게 스윕. 이건 `trajectory_studio._sweep_reachability` 와 동일한
    로직(wrist_3 lock)이라 GLNS 대비 보수적(과소평가)이므로, **상대적 top-K pre-filter** 로만 쓴다.
  Stage 2 (비쌈) — 살아남은 배치에만 `solve_glns_path.py` 를 Studio 기본 파라미터로 subprocess
    실행. 결과 h5(metadata + per-component reconfig)를 읽어 채점.

채점(도달 우선 lexicographic): `(placement_drop, base, reconfigs, wrist, big_flips)` 오름차순.
  - placement_drop = |A| - (그 배치가 GLNS 로 실제 방문한 viewpoint 수)
    A = 그리드 어느 셀에서든 ≥1회 도달된 viewpoint 집합(achievable). 본질적 미도달(N-|A|)은
    모든 배치에 동일 상쇄되어 랭킹에 영향 없음.
  → 달성 가능한 최대 coverage 우선, 그 중 base reconfig 최소.

산출물: data/{object}/placement_sweep/{summary.csv, summary.json, glns_result_*.h5, heatmap_z*.png}.
최적 배치 h5 는 `trajectory_studio.py --result <h5>` 로 바로 시각 검증 가능.

사용법:
    uv run --no-sync scripts/tools/optimize_placement.py --object sample
    uv run --no-sync scripts/tools/optimize_placement.py            # 4개 물체 전부
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT / "core"))
sys.path.insert(0, str(SCRIPTS_ROOT / "apps"))

from common import config  # noqa: E402
from common.glns_utils import read_result_hdf5  # noqa: E402
import plan_trajectory as PT  # noqa: E402  (heavy: curobo import)

DATA_ROOT = PROJECT_ROOT / "data"

# trajectory_studio "Solve GLNS" 기본 파라미터 (trajectory_studio.py:680-705 와 동일).
GLNS_DEFAULT_ARGS = [
    "--delaunay-expand-hops", "2",
    "--roll-augment",
    "--tilt-augment", "--tilt-angles-deg", "5", "10", "--tilt-azimuths", "8",
    "--max-candidates-per-viewpoint", "32",
    "--num-seeds", "32", "--ik-batch-size", "128",
]
BIG_FLIP_DEG = 60.0  # Studio _compute_metrics 와 동일한 큰 flip 판정

# config.TARGET_OBJECT 기본값(= OBJECT_PLACEMENTS 항목 없는 물체의 그리드 중심).
# 물체 루프가 이 전역을 셀마다 mutate 하므로, import 시점의 pristine 값을 잡아두고
# 항목 없는 물체에서는 직전 물체가 남긴 오염된 값 대신 이 기본값으로 복원한다.
_DEFAULT_POS = np.asarray(config.TARGET_OBJECT["position"], dtype=np.float64).copy()
_DEFAULT_QUAT = np.asarray(config.TARGET_OBJECT["rotation"], dtype=np.float64).copy()


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 준비
# ─────────────────────────────────────────────────────────────────────────────
def discover_objects(data_root: Path) -> list[str]:
    return [p.parent.parent.name for p in sorted(data_root.glob("*/mesh/source.obj"))]


def find_viewpoints(object_name: str) -> Path | None:
    """data/{object}/viewpoint/*/viewpoints*.h5 중 하나(가장 그럴듯한 것)를 고른다."""
    base = DATA_ROOT / object_name / "viewpoint"
    cands = sorted(base.glob("*/viewpoints*.h5"))
    if not cands:
        return None
    # coacd 결과를 우선(파이프라인 기본), 없으면 첫 번째.
    for p in cands:
        if "coacd" in p.name:
            return p
    return cands[0]


def load_object_viewpoints(vp_path: Path):
    """positions/normals/wd_m 로드(path_order 로 정렬 — 파이프라인/Studio 동일)."""
    positions, normals, path_order, _cluster, wd_m = PT.load_viewpoints(vp_path)
    if path_order is not None:
        order = np.argsort(path_order)
        positions, normals = positions[order], normals[order]
    return positions, normals, wd_m


def table_top_z() -> float:
    return float(config.TABLE["position"][2] + config.TABLE["dimensions"][2] / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — 빠른 IK 도달률 사전필터
# ─────────────────────────────────────────────────────────────────────────────
def sweep_reachability(backend, positions, normals, wd_m) -> np.ndarray:
    """현재 config.TARGET_OBJECT pose 에서 모든 viewpoint 의 collision-free 도달 여부 (N,) bool.

    trajectory_studio._sweep_reachability (trajectory_studio.py:577-590) 와 동일 로직.
    """
    world_poses = PT.build_camera_poses(positions, normals, wd_m)
    from ik_backend import _wxyz_from_matrix  # noqa: PLC0415

    N = len(world_poses)
    free = np.zeros(N, dtype=bool)
    for i in range(N):
        T = world_poses[i]
        reps, colliding = backend.solve_reps(T[:3, 3], _wxyz_from_matrix(T[:3, :3]))
        free[i] = len(reps) > 0 and bool((~colliding).any())
    return free


def stage1_grid_sweep(object_name, positions, normals, wd_m, grid, base_quat, log):
    """그리드 각 셀의 Stage-1 도달률을 잰다. returns list of cell dicts, achievable mask."""
    from ik_backend import IKBackend  # noqa: PLC0415  (heavy: curobo)

    log(f"[Stage 1] {object_name}: {len(grid)} cells × {len(positions)} vp IK sweep…")
    backend = IKBackend(object_name)
    z_floor = table_top_z() + 0.02  # support 높이 > 0 보장(sync_support_to_target 예외 회피)

    cells = []
    reach_union = np.zeros(len(positions), dtype=bool)
    for gi, (gx, gy, gz, pos) in enumerate(grid):
        if float(pos[2]) <= z_floor:
            log(f"  cell {gi:3d} pos={np.round(pos,3).tolist()} z<=table → skip")
            cells.append({"gi": gi, "gx": gx, "gy": gy, "gz": gz, "pos": pos,
                          "reachable": -1, "mask": None})
            continue
        config.TARGET_OBJECT["position"] = np.asarray(pos, dtype=np.float64)
        config.TARGET_OBJECT["rotation"] = np.asarray(base_quat, dtype=np.float64)
        try:
            config.sync_support_to_target()
            backend.rebuild_world()
            mask = sweep_reachability(backend, positions, normals, wd_m)
        except Exception as exc:  # noqa: BLE001
            log(f"  cell {gi:3d} pos={np.round(pos,3).tolist()} FAILED: {exc}")
            cells.append({"gi": gi, "gx": gx, "gy": gy, "gz": gz, "pos": pos,
                          "reachable": -1, "mask": None})
            continue
        reach_union |= mask
        cells.append({"gi": gi, "gx": gx, "gy": gy, "gz": gz, "pos": pos,
                      "reachable": int(mask.sum()), "mask": mask})
        log(f"  cell {gi:3d} pos={np.round(pos,3).tolist()} "
            f"reachable {int(mask.sum())}/{len(positions)}")

    # in-process cuRobo 반납 → Stage-2 subprocess co-residency 완화 (Studio 와 동일 패턴)
    del backend
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return cells, reach_union


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — GLNS solve (subprocess) + 채점
# ─────────────────────────────────────────────────────────────────────────────
def run_glns_solve(object_name, vp_path, pos, quat, out_h5, log_path, timeout_s):
    """solve_glns_path.py 를 Studio 기본 파라미터로 실행. returns (rc, wall_seconds)."""
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "--no-sync", "scripts/core/solve_glns_path.py",
        "--object", object_name, "--viewpoints", str(vp_path),
        "--object-position", *(f"{v:.6f}" for v in pos),
        "--object-quat", *(f"{v:.6f}" for v in quat),
        "--glns-seed", "42",
        "--output", str(out_h5),
        *GLNS_DEFAULT_ARGS,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.perf_counter()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), env=env,
            stdout=logf, stderr=subprocess.STDOUT, text=True,
            timeout=timeout_s, check=False,
        )
    return proc.returncode, time.perf_counter() - t0


def compute_glns_metrics(result) -> dict:
    """GLNS 결과 dict → 채점 지표. Studio _compute_metrics (trajectory_studio.py:888-904) 확장.

    visited = solved component 멤버 합(= 실제 GLNS 방문 coverage). component 실패 시 그 멤버 미방문.
    """
    reach = np.asarray(result["reachable_mask"], dtype=bool)
    N = len(reach)
    reconfigs = base = wrist = flips = visited = 0
    solved = 0
    for c in result["components"]:
        if c["status"] != "solved":
            continue
        solved += 1
        a = c["attrs"]
        reconfigs += int(a.get("num_reconfigurations", 0))
        base += int(a.get("num_reconfigurations_base", 0))
        wrist += int(a.get("num_reconfigurations_wrist", 0))
        visited += int(len(np.asarray(c["members"])))
        sel = c.get("selected_joints")
        if sel is None or len(sel) < 2:
            continue
        d = np.degrees(np.max(np.abs(np.diff(np.asarray(sel), axis=0)), axis=1))
        flips += int((d > BIG_FLIP_DEG).sum())
    meta = result["metadata"]
    return {
        "N": N,
        "reachable": int(reach.sum()),
        "dropped_unreachable": int((~reach).sum()),
        "num_components": int(_meta_int(meta, "num_components", len(result["components"]))),
        "solved_components": solved,
        "visited": visited,
        "reconfigs": reconfigs,
        "base": base,
        "wrist": wrist,
        "flips": flips,
    }


def _meta_int(meta, key, default=0):
    v = meta.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# 그리드 & 리포트
# ─────────────────────────────────────────────────────────────────────────────
def build_grid(center, xy_extent, xy_step, z_offsets):
    """center(3,) 주변 XY(±extent step) × Z(center_z + offsets) 그리드.

    returns list of (gx, gy, gz, pos). gx/gy/gz 는 격자 인덱스(히트맵용).
    """
    n = int(round(xy_extent / xy_step))
    offs = np.round(np.arange(-n, n + 1) * xy_step, 6)  # 대칭, 0 포함
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    grid = []
    for gz, dz in enumerate(z_offsets):
        for gx, dx in enumerate(offs):
            for gy, dy in enumerate(offs):
                pos = np.array([cx + dx, cy + dy, cz + float(dz)], dtype=np.float64)
                grid.append((gx, gy, gz, pos))
    return grid, offs, list(z_offsets)


def save_heatmaps(out_dir, object_name, cells, results_by_gi, offs, z_offsets, center):
    """Stage-1 도달률 + base reconfig 히트맵(Z 레벨별) PNG 저장. matplotlib 없으면 skip."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return
    m = len(offs)
    if m < 2:  # 1×1 그리드는 히트맵 의미 없음(singular lims 경고 회피)
        return
    for gz in range(len(z_offsets)):
        reach = np.full((m, m), np.nan)
        basev = np.full((m, m), np.nan)
        for c in cells:
            if c["gz"] != gz:
                continue
            if c["reachable"] >= 0:
                reach[c["gx"], c["gy"]] = c["reachable"]
            r = results_by_gi.get(c["gi"])
            if r is not None:
                basev[c["gx"], c["gy"]] = r["base"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        extent = [offs[0], offs[-1], offs[0], offs[-1]]
        for ax, data, title, cmap in (
            (axes[0], reach.T, "Stage-1 reachable", "viridis"),
            (axes[1], basev.T, "GLNS base reconfig", "viridis_r"),
        ):
            im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, aspect="equal")
            ax.set_title(title)
            ax.set_xlabel("dx (m)")
            ax.set_ylabel("dy (m)")
            fig.colorbar(im, ax=ax, fraction=0.046)
        z_abs = float(center[2]) + float(z_offsets[gz])
        fig.suptitle(f"{object_name}  z={z_abs:.3f}m (offset {z_offsets[gz]:+.3f})")
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_z{gz}.png", dpi=110)
        plt.close(fig)


def rank_key(row):
    """도달 우선 lexicographic (min-sort): visited↓ → base↑ → reconfigs↑ → wrist↑ → flips↑.

    coverage 는 순수 GLNS 실측 visited(=solved component 멤버 합)로만 판단한다. Stage-1 IK 는
    wrist_3 lock 이라 GLNS(roll/tilt augment, wrist free)보다 도달률을 과소평가하므로 점수에 쓰지 않고
    top-K 사전필터로만 쓴다.
    """
    return (-row["visited"], row["base"], row["reconfigs"], row["wrist"], row["flips"])


# ─────────────────────────────────────────────────────────────────────────────
# 물체 1개 처리
# ─────────────────────────────────────────────────────────────────────────────
def optimize_object(object_name, args, log):
    vp_path = find_viewpoints(object_name)
    if vp_path is None:
        log(f"[skip] {object_name}: viewpoint h5 없음")
        return None
    positions, normals, wd_m = load_object_viewpoints(vp_path)
    N = len(positions)

    # 중심 배치 = per-object 기본값(없으면 pristine TARGET_OBJECT 기본), 회전 고정.
    # apply_object_placement 가 False(항목 없음)면 직전 물체의 오염된 전역이 남으므로 명시적 복원.
    if not config.apply_object_placement(object_name):
        config.TARGET_OBJECT["position"] = _DEFAULT_POS.copy()
        config.TARGET_OBJECT["rotation"] = _DEFAULT_QUAT.copy()
        config.sync_support_to_target()
    center = np.asarray(config.TARGET_OBJECT["position"], dtype=np.float64).copy()
    base_quat = np.asarray(config.TARGET_OBJECT["rotation"], dtype=np.float64).copy()

    grid, offs, z_offsets = build_grid(center, args.xy_extent, args.xy_step, args.z_offsets)
    out_dir = DATA_ROOT / object_name / "placement_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n=== {object_name}: N={N} vp, center={np.round(center,3).tolist()}, "
        f"quat={np.round(base_quat,3).tolist()}, grid={len(grid)} cells "
        f"(XY {len(offs)}×{len(offs)} × Z {len(z_offsets)}) → {out_dir} ===")

    # Stage 1
    cells, reach_union = stage1_grid_sweep(
        object_name, positions, normals, wd_m, grid, base_quat, log)
    A = int(reach_union.sum())
    inherent_unreach = N - A
    valid = [c for c in cells if c["reachable"] >= 0]
    if not valid:
        log(f"[skip] {object_name}: 유효 셀 없음")
        return None
    max_reach = max(c["reachable"] for c in valid)

    # 생존 선택: reachable >= max - tau, top-K
    survivors = [c for c in valid if c["reachable"] >= max_reach - args.reach_tau]
    survivors.sort(key=lambda c: -c["reachable"])
    survivors = survivors[: args.max_glns]
    log(f"[Stage 1 done] achievable |A|={A}/{N} (inherent unreachable={inherent_unreach}), "
        f"grid max reachable={max_reach}, survivors={len(survivors)}/{len(valid)} "
        f"(tau={args.reach_tau}, cap={args.max_glns})")

    # Stage 2 — 생존 배치에만 GLNS
    log(f"[Stage 2] {object_name}: GLNS solve × {len(survivors)}…")
    rows = []
    results_by_gi = {}
    for si, c in enumerate(survivors):
        pos = c["pos"]
        out_h5 = out_dir / f"glns_result_{c['gi']:03d}.h5"
        log_path = out_dir / f"solve_{c['gi']:03d}.log"
        try:
            rc, secs = run_glns_solve(
                object_name, vp_path, pos, base_quat, out_h5, log_path, args.solve_timeout)
        except subprocess.TimeoutExpired:
            log(f"  [{si+1}/{len(survivors)}] cell {c['gi']} pos={np.round(pos,3).tolist()} "
                f"TIMEOUT(>{args.solve_timeout}s) → skip")
            continue
        if not out_h5.exists():
            log(f"  [{si+1}/{len(survivors)}] cell {c['gi']} rc={rc} no h5 → skip "
                f"(see {log_path.name})")
            continue
        try:
            result = read_result_hdf5(out_h5)
            m = compute_glns_metrics(result)
        except Exception as exc:  # noqa: BLE001
            log(f"  [{si+1}/{len(survivors)}] cell {c['gi']} read FAILED: {exc}")
            continue
        row = {
            "gi": c["gi"], "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "stage1_reachable": c["reachable"],
            "glns_reachable": m["reachable"], "visited": m["visited"],
            "uncovered": m["N"] - m["visited"],
            "base": m["base"], "reconfigs": m["reconfigs"],
            "wrist": m["wrist"], "flips": m["flips"],
            "solved_components": m["solved_components"], "num_components": m["num_components"],
            "rc": rc, "seconds": round(secs, 1),
            "glns_h5": str(out_h5.relative_to(PROJECT_ROOT)),
        }
        rows.append(row)
        results_by_gi[c["gi"]] = row
        log(f"  [{si+1}/{len(survivors)}] cell {c['gi']} pos={np.round(pos,3).tolist()} "
            f"visited={m['visited']}/{m['N']} base={m['base']} reconfigs={m['reconfigs']} "
            f"wrist={m['wrist']} flips={m['flips']} ({secs:.0f}s, rc={rc})")

    if not rows:
        log(f"[fail] {object_name}: GLNS 성공 배치 없음")
        return None

    rows.sort(key=rank_key)
    best_visited = rows[0]["visited"]
    for r in rows:
        r["gap_to_best"] = best_visited - r["visited"]  # 최고 coverage 대비 부족분
    best = rows[0]

    # summary 저장
    summary = {
        "object": object_name, "N": N, "achievable_A": A,
        "inherent_unreachable": inherent_unreach,
        "center": center.tolist(), "quat_wxyz": base_quat.tolist(),
        "viewpoints": str(vp_path.relative_to(PROJECT_ROOT)),
        "grid": {"xy_extent": args.xy_extent, "xy_step": args.xy_step,
                 "z_offsets": list(args.z_offsets)},
        "best": best, "rows": rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    fields = ["gi", "x", "y", "z", "stage1_reachable", "glns_reachable", "visited",
              "uncovered", "gap_to_best", "base", "reconfigs", "wrist", "flips",
              "solved_components", "num_components", "rc", "seconds", "glns_h5"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    save_heatmaps(out_dir, object_name, cells, results_by_gi, offs, z_offsets, center)

    # 콘솔 랭킹 표
    _print_ranking(object_name, rows, A, N, center, log)
    return summary


def _print_ranking(object_name, rows, A, N, center, log, top=10):
    log(f"\n──── {object_name}: 랭킹 (N={N}, Stage-1 achievable≈{A}; coverage=GLNS visited) ────")
    log(f"{'rank':>4} {'dx':>7} {'dy':>7} {'dz':>7} {'visit':>7} {'uncov':>5} "
        f"{'base':>5} {'recfg':>6} {'wrist':>6} {'flips':>6} {'sec':>5}")
    for i, r in enumerate(rows[:top]):
        dx, dy, dz = r["x"]-center[0], r["y"]-center[1], r["z"]-center[2]
        log(f"{i+1:>4} {dx:>+7.3f} {dy:>+7.3f} {dz:>+7.3f} {r['visited']:>4}/{N:<2} "
            f"{r['uncovered']:>5} {r['base']:>5} {r['reconfigs']:>6} "
            f"{r['wrist']:>6} {r['flips']:>6} {r['seconds']:>5.0f}")
    b = rows[0]
    # 기준선(중심 셀) 비교
    base_row = next((r for r in rows if abs(r["x"]-center[0]) < 1e-6
                     and abs(r["y"]-center[1]) < 1e-6 and abs(r["z"]-center[2]) < 1e-6), None)
    log(f"★ best {object_name}: pos=[{b['x']:.3f}, {b['y']:.3f}, {b['z']:.3f}] "
        f"(offset [{b['x']-center[0]:+.3f}, {b['y']-center[1]:+.3f}, {b['z']-center[2]:+.3f}]) "
        f"visited={b['visited']}/{N} base={b['base']} reconfigs={b['reconfigs']} → {b['glns_h5']}")
    if base_row is not None:
        log(f"  baseline(center): visited={base_row['visited']}/{N} base={base_row['base']} "
            f"reconfigs={base_row['reconfigs']}  →  개선 visited {base_row['visited']}→{b['visited']}, "
            f"base {base_row['base']}→{b['base']}, reconfigs {base_row['reconfigs']}→{b['reconfigs']}")
    else:
        log("  baseline(center): GLNS 미평가(생존 아님)")
    # 최소-reconfig 대안(coverage 무시, 순수 reconfig 최소) — 승자와 다르면 함께 안내.
    alt = min(rows, key=lambda r: (r["base"], r["reconfigs"], r["wrist"], r["flips"], -r["visited"]))
    if alt["gi"] != b["gi"]:
        log(f"  ↳ 최소-reconfig 대안: pos=[{alt['x']:.3f}, {alt['y']:.3f}, {alt['z']:.3f}] "
            f"(offset [{alt['x']-center[0]:+.3f}, {alt['y']-center[1]:+.3f}, {alt['z']-center[2]:+.3f}]) "
            f"visited={alt['visited']}/{N} base={alt['base']} reconfigs={alt['reconfigs']} "
            f"→ {alt['glns_h5']}  (coverage {b['visited']-alt['visited']}개 적지만 reconfig 우선 시)")


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="물체 최적 배치 탐색 (GLNS reconfig 기준, headless)")
    p.add_argument("--object", type=str, default=None,
                   help="대상 물체(생략 시 mesh 있는 4개 전부)")
    p.add_argument("--xy-extent", type=float, default=0.15, help="XY 스윕 반경 m (default 0.15)")
    p.add_argument("--xy-step", type=float, default=0.075, help="XY 스윕 간격 m (default 0.075 → 5×5)")
    p.add_argument("--z-offsets", type=float, nargs="+", default=[0.0, 0.05, 0.10],
                   help="중심 z 대비 오프셋 m (default 0 0.05 0.10)")
    p.add_argument("--reach-tau", type=int, default=3,
                   help="생존 기준: Stage-1 reachable >= 그리드최대-tau (default 3)")
    p.add_argument("--max-glns", type=int, default=20,
                   help="GLNS 실행할 최대 배치 수(top-K, default 20)")
    p.add_argument("--solve-timeout", type=float, default=600.0,
                   help="배치당 solve_glns_path 최대 대기 초 (default 600)")
    return p.parse_args()


def main():
    args = parse_args()
    all_objects = discover_objects(DATA_ROOT)
    if not all_objects:
        raise SystemExit(f"No objects with mesh/source.obj under {DATA_ROOT}")
    if args.object:
        if args.object not in all_objects:
            raise SystemExit(f"'{args.object}' 없음. 가능: {all_objects}")
        objects = [args.object]
    else:
        objects = all_objects

    def log(msg):
        print(msg, flush=True)

    log(f"[optimize_placement] objects={objects}")
    summaries = []
    for obj in objects:
        try:
            s = optimize_object(obj, args, log)
        except Exception as exc:  # noqa: BLE001
            import traceback
            log(f"[error] {obj}: {exc}\n{traceback.format_exc()}")
            continue
        if s is not None:
            summaries.append(s)

    if summaries:
        log("\n════════ 요약 (물체별 최적 배치) ════════")
        for s in summaries:
            b = s["best"]
            c = s["center"]
            log(f"  {s['object']:16s} pos=[{b['x']:.3f}, {b['y']:.3f}, {b['z']:.3f}] "
                f"(offset [{b['x']-c[0]:+.3f}, {b['y']-c[1]:+.3f}, {b['z']-c[2]:+.3f}])  "
                f"visited={b['visited']}/{s['N']}  base={b['base']}  "
                f"reconfigs={b['reconfigs']}")


if __name__ == "__main__":
    main()

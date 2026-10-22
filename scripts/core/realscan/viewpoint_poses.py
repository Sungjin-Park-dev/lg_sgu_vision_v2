#!/usr/bin/env python3
"""검사 viewpoint → PiPER 자세 — viewpoints.h5 의 표면 점마다 카메라를 작업 거리만큼 띄워 IK 를 푼다 (pose_tour YAML).

viewpoint(core/viewpoint 가 복원 메시로 만든 것)는 메시 좌표다. --mesh-frame 의 board_T_object 로 보드 작업
좌표(보드 중심, x 긴 변, z 위)로 옮기고, --work-frame(board_check.py --scan-board --save-frame 으로 잰 base 위치)
으로 base 에 놓는다. 카메라(depth)는 표면 점에서 법선 방향 WD(h5 의 working_distance_mm)에 두고 그 점을 본다.
광축 둘레 roll 은 viewpoint 가 정하지 않으므로 --rolls 개를 돌려 본다. 남기는 조건은 scan_poses 와 같다:
  관절 한계(여유) · 손목 뻗음 ≤ --max-reach · 벌린 손가락이 물체·받침대 상자와 보드면에서 --finger-clear 이상 ·
  보이는 보드 코너 ≥ --min-corners (보드 추적으로 찍을 때 필요).
통과한 후보 중 코너가 많은 것(같으면 손가락 여유가 큰 것)을 고른다.

회전판: --work-frame 을 판 각도마다 하나씩(돌릴 때마다 잰 JSON) 주면 viewpoint 마다 어느 각도에서 되는지 표로
보이고, 각도마다 YAML 을 쓴다(그 각도에서 통과한 viewpoint 를 관절 이동이 짧은 순서로). 한 viewpoint 가 여러
각도에서 되면 모든 YAML 에 들어간다 — 찍을 때 고른다.

    python scripts/core/realscan/viewpoint_poses.py --viewpoints data/tomato/viewpoint/20/viewpoints.h5 \\
        --mesh-frame data/tomato/mesh/recon_frame.json \\
        --work-frame output/piper_scan_poses/tomato_frame.json output/piper_scan_poses/tomato_tt_2.json \\
        --keep-out 0 0 0.0205 0.05 0.05 0.041 --views-from data/tomato/scan/20261020_123514_978614 \\
        --fingers-from data/blue_cube/scan/20261020_020042 --out output/piper_scan_poses/tomato_vp.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from core.realscan.calib_poses import FastFK, LIMIT_MARGIN_DEG, order_by_motion, solve_ik, wrist_reach  # noqa: E402
from core.realscan.capture import transform  # noqa: E402
from core.realscan.piper import PiperFK  # noqa: E402
from core.realscan.scan_poses import (Scene, board_corners, load_fingers, load_seeds, load_work_frame,  # noqa: E402
                                      obstacle_for_tour, view_pose)


def load_viewpoints(path):
    """viewpoints.h5 → (점 (N,3), 법선 (N,3), 작업 거리 [m]) — 메시 좌표."""
    import h5py
    with h5py.File(path, "r") as f:
        return (np.asarray(f["viewpoints/positions"], float), np.asarray(f["viewpoints/normals"], float),
                float(f["metadata/camera_spec"].attrs["working_distance_mm"]) / 1000)


def roll_poses(point, normal, wd, rolls):
    """표면 점을 법선 방향 wd 에서 보는 카메라 자세(작업 좌표) — 광축 둘레 roll 을 360/rolls° 씩."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = np.cross(n, ref)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    return [view_pose(n, np.cos(t) * a + np.sin(t) * b, np.asarray(point, float), wd)
            for t in 2 * np.pi * np.arange(rolls) / rolls]


def plan_viewpoints(ffk, seeds, base_T_world, tool_T_depth, scene, points, normals, wds, *, rolls,
                    min_corners, finger_clear, max_reach, lo, hi, finger_low=None, max_move_deg=60,
                    seeds_per_view=4):
    """viewpoint 마다 → 통과한 가장 좋은 후보(dict, ok=True) 또는 가장 덜 어긋난 후보(ok=False, reason).

    wds: 작업 거리 후보 [m] — 앞의 것부터, 처음 통과한 거리에서 멈춘다.
    finger_low: 손가락 최저점의 보드면 위 높이 하한 [m] (기본 = finger_clear).
    max_move_deg: 가장 가까운 시드에서의 관절 변화 상한 — 다녀 본 팔 모양에서 멀리 뒤집힌 해(손목 뒤집힘)를 버린다.
    2026-10-20 이 제한 없이 고른 손목 뒤집힌 자세로 가다 관절 1 이 115° 돌며 카메라 쪽이 팔에 부딪혔다.
    """
    finger_low = finger_clear if finger_low is None else finger_low
    world_T_base = np.linalg.inv(base_T_world)
    cams = [world_T_base @ ffk.pose(q) @ tool_T_depth for q in seeds]       # 작업 좌표의 시드 카메라
    out = []
    for number, (p, n) in enumerate(zip(points, normals), 1):
        n = n / np.linalg.norm(n)
        # 시드: 그 시드 카메라가 이 표면 점을 법선 쪽에서 보던 것부터
        order = np.argsort([-((c[:3, 3] - p) / np.linalg.norm(c[:3, 3] - p)) @ n for c in cams])[:seeds_per_view]
        best, best_fail = None, None
        for wd, T in ((wd, T) for wd in wds for T in roll_poses(p, n, wd, rolls)):
            if best is not None and wd != best["wd_m"]:
                break
            q = solve_ik(ffk, base_T_world @ T @ np.linalg.inv(tool_T_depth), [seeds[i] for i in order], lo, hi)
            if q is None:
                continue
            move = float(np.degrees(min(np.max(np.abs(q - seeds[i])) for i in order)))
            clear, low = scene.finger_clearance(T)
            e = {"number": number, "q": q, "wd_m": wd, "corners": scene.visible_corners(T), "finger_clear_m": clear,
                 "finger_low_m": low, "reach_m": wrist_reach(ffk.frames(q)), "camera_z_m": float(T[2, 3]),
                 "elev_deg": float(np.degrees(np.arcsin(np.clip(n[2], -1, 1)))), "move_deg": move}
            bad = []
            if move > max_move_deg:
                bad.append(f"관절 변화 {move:.0f}°")
            if e["reach_m"] > max_reach:
                bad.append(f"뻗음 {e['reach_m'] * 100:.1f}cm")
            if clear < finger_clear:
                bad.append(f"손가락 {clear * 100:.1f}cm")
            if low < finger_low:
                bad.append(f"손가락 높이 {low * 100:.1f}cm")
            if e["corners"] < min_corners:
                bad.append(f"코너 {e['corners']}")
            e["ok"], e["reason"] = not bad, ", ".join(bad)
            if e["ok"]:
                if best is None or (e["corners"], e["finger_clear_m"]) > (best["corners"], best["finger_clear_m"]):
                    best = e
            elif best_fail is None or len(bad) < len(best_fail["reason"].split(", ")):
                best_fail = e
        out.append(best or best_fail or {"number": number, "ok": False, "reason": "IK 해 없음",
                                         "elev_deg": float(np.degrees(np.arcsin(np.clip(n[2], -1, 1))))})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--viewpoints", required=True, help="viewpoints.h5 (core/viewpoint/cli.py 산출물)")
    p.add_argument("--mesh-frame", required=True,
                   help="메시 좌표 → 보드 작업 좌표 JSON (board_T_object) — 복원 메시를 내보낼 때 남긴 것")
    p.add_argument("--work-frame", nargs="+", required=True,
                   help="보드 작업 좌표의 base 위치 JSON (board_check --save-frame) — 판 각도마다 하나")
    p.add_argument("--keep-out", type=float, nargs=6, action="append", default=[],
                   metavar=("CX", "CY", "CZ", "SX", "SY", "SZ"), help="손가락이 피하고 코너를 가리는 상자(받침대 등)")
    p.add_argument("--object-margin", type=float, default=0.003, help="메시 상자에 더할 여유 [m]")
    p.add_argument("--object-shift", type=float, nargs=3, default=[0, 0, 0], metavar=("DX", "DY", "DZ"),
                   help="물체를 다시 놓아 복원 때와 어긋난 만큼 [m, 보드 작업 좌표] — viewpoint 와 물체 상자를 같이 옮긴다")
    p.add_argument("--views-from", nargs="+", help="IK 시드: 촬영 run 폴더 (capture.json 관절값)")
    p.add_argument("--poses", nargs="+", help="IK 시드: 자세 YAML (poses_deg)")
    p.add_argument("--fingers-from", help="손가락이 찍힌 run 폴더(벌린 그리퍼) — 없으면 손가락 검사를 건너뛴다")
    p.add_argument("--working-distance", type=float, nargs="+",
                   help="작업 거리 후보 [m] — viewpoint 마다 앞의 것부터 처음 통과한 거리 (기본: h5 의 working_distance_mm)")
    p.add_argument("--rolls", type=int, default=12, help="광축 둘레 roll 후보 수")
    p.add_argument("--min-corners", type=int, default=20, help="계산상 보이는 보드 코너 최소 (실제 ~55–60%%)")
    p.add_argument("--finger-clear", type=float, default=0.025, help="손가락 ↔ 물체·받침대 상자 최소 거리 [m]")
    p.add_argument("--finger-low", type=float, help="손가락 최저점의 보드면 위 높이 하한 [m] (기본 = --finger-clear)")
    p.add_argument("--max-move-deg", type=float, default=60, help="가장 가까운 시드에서의 관절 변화 상한 [deg]")
    p.add_argument("--max-reach", type=float, default=0.228,
                   help="joint2 축 ↔ 손목 수평 거리 상한 [m] (23.9 cm 자세는 떨려서 정지 판정에 실패했다)")
    p.add_argument("--out", required=True, help="pose_tour --poses YAML (판 각도가 여럿이면 _1, _2 … 를 붙인다)")
    p.add_argument("--config", default=str(ROOT / "workcell/scans/piper_d405.yaml"))
    p.add_argument("--calibration", default=str(ROOT / "workcell/calibration/piper_d405.yaml"))
    args = p.parse_args()
    if not (args.views_from or args.poses):
        p.error("IK 시드가 필요합니다: --views-from 또는 --poses")

    import trimesh
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    cal = yaml.safe_load(Path(args.calibration).read_text())
    ffk = FastFK(PiperFK(ROOT / cfg["urdf"], cfg["base_frame"], cfg["tool_frame"]))
    lo = ffk.lower + np.radians(LIMIT_MARGIN_DEG)
    hi = ffk.upper - np.radians(LIMIT_MARGIN_DEG)
    tool_T_depth = transform(cal["tool_T_depth"])
    color = cal["identity"]["color"]
    size = (int(color["width"]), int(color["height"]))

    vp_path = Path(args.viewpoints)
    pts, nrm, wd = load_viewpoints(vp_path if vp_path.is_absolute() else ROOT / vp_path)
    wds = args.working_distance or [wd]
    frame_path = Path(args.mesh_frame)
    frame = json.loads((frame_path if frame_path.is_absolute() else ROOT / frame_path).read_text())
    board_T_object = transform(frame["board_T_object"])
    board_T_object[:3, 3] += args.object_shift
    points = pts @ board_T_object[:3, :3].T + board_T_object[:3, 3]
    normals = nrm @ board_T_object[:3, :3].T
    # 물체 상자: 메시 꼭짓점을 보드 좌표로 옮긴 범위
    mesh_path = ROOT / frame["source_run"] / "recon.ply" if "source_run" in frame else None
    mesh = trimesh.load(str(mesh_path), force="mesh") if mesh_path and mesh_path.exists() else None
    if mesh is None:
        raise SystemExit(f"물체 메시를 찾지 못했습니다: {mesh_path}")
    lo_b, hi_b = mesh.bounds          # recon.ply 는 이미 보드 작업 좌표
    obj_box = ((lo_b + hi_b) / 2 + args.object_shift, hi_b - lo_b + 2 * args.object_margin)
    boxes = [obj_box] + [(b[:3], b[3:]) for b in args.keep_out]

    seeds, sources = [], []
    for kind in ("views_from", "poses"):
        if getattr(args, kind):
            ns = argparse.Namespace(views_from=None, poses=None)
            setattr(ns, kind, getattr(args, kind))
            s, name = load_seeds(ns)
            seeds += s
            sources.append(name)
    fingers = mask = None
    if args.fingers_from:
        fingers, mask = load_fingers(args.fingers_from, size)
    else:
        print("⚠️ --fingers-from 이 없어 손가락 충돌·가림을 검사하지 않습니다")
    scene = Scene(board_corners(cfg["scan_board"]), boxes, fingers, mask,
                  np.asarray(color["K"], float).reshape(3, 3), size, cal["identity"]["color_T_depth"])
    print(f"viewpoint {len(points)}개 · 작업 거리 {'/'.join(f'{w * 1000:.0f}' for w in wds)} mm · 물체 상자 "
          f"{np.round(obj_box[1] * 1000).astype(int).tolist()} mm · 시드 {len(seeds)}개 · roll {args.rolls}개")

    results = []
    for wf in args.work_frame:
        base_T_world = load_work_frame(wf)
        results.append(plan_viewpoints(ffk, seeds, base_T_world, tool_T_depth, scene, points, normals, wds,
                                       rolls=args.rolls, min_corners=args.min_corners,
                                       finger_clear=args.finger_clear, max_reach=args.max_reach, lo=lo, hi=hi,
                                       finger_low=args.finger_low, max_move_deg=args.max_move_deg))
    labels = [Path(wf).stem for wf in args.work_frame]
    print("\n번호 높이각 | " + " | ".join(f"{s:>22s}" for s in labels))
    for i in range(len(points)):
        cells = []
        for rows in results:
            r = rows[i]
            cell = "✗ " + r["reason"] if not r["ok"] else f"✓ {r['wd_m'] * 1000:.0f}mm 코너 {r['corners']}" + (
                f" 손가락 {r['finger_clear_m'] * 100:.1f}cm" if np.isfinite(r["finger_clear_m"]) else "")
            cells.append(f"{cell:>22s}")
        print(f"#{i + 1:2d}  {results[0][i]['elev_deg']:5.0f}° | " + " | ".join(cells))
    covered = [i for i in range(len(points)) if any(rows[i]["ok"] for rows in results)]
    print(f"\n각도별 통과: " + ", ".join(f"{s} {sum(r['ok'] for r in rows)}" for s, rows in zip(labels, results))
          + f" · 어느 각도에서든 {len(covered)}/{len(points)}")
    missing = [i + 1 for i in range(len(points)) if i not in covered]
    if missing:
        print(f"어느 각도에서도 안 되는 viewpoint: {missing}")

    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    for k, (wf, rows) in enumerate(zip(args.work_frame, results), 1):
        path = out if len(results) == 1 else out.with_name(f"{out.stem}_{k}{out.suffix}")
        kept = [r for r in rows if r["ok"]]
        if kept:
            kept = order_by_motion(kept, max(kept, key=lambda r: r["camera_z_m"])["q"])
        obstacle = obstacle_for_tour(load_work_frame(wf), boxes)
        lines = [f"# viewpoint_poses.py — {args.viewpoints} (작업 거리 후보 {'/'.join(f'{w * 1000:.0f}' for w in wds)} mm), "
                 f"작업 좌표 {wf}",
                 f"# 시드 {', '.join(sources)} · 받침대 등 {args.keep_out}",
                 f"# pose_tour 충돌 상자(base): --obstacle {' '.join(f'{v:.3f}' for v in obstacle)}",
                 "# 각 줄: 관절값[deg] — 주석: viewpoint 번호·작업 거리·법선 높이각·계산 코너·손가락 여유·손목 뻗음",
                 "poses_deg:"]
        for r in kept:
            lines.append(f"- {yaml.safe_dump([round(float(x), 3) for x in np.degrees(r['q'])], default_flow_style=True).strip()}"
                         f"  # vp{r['number']} 거리 {r['wd_m'] * 1000:.0f}mm 높이각 {r['elev_deg']:.0f}° 코너 {r['corners']} "
                         + (f"손가락 {r['finger_clear_m'] * 100:.1f}cm " if np.isfinite(r["finger_clear_m"]) else "")
                         + f"뻗음 {r['reach_m'] * 100:.1f}cm")
        path.write_text("\n".join(lines) + "\n")
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"{labels[k - 1]}: {len(kept)}개 → {rel} · pose_tour --obstacle {' '.join(f'{v:.3f}' for v in obstacle)}")


if __name__ == "__main__":
    main()

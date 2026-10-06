#!/usr/bin/env python3
"""관심 영역(구) 둘레에 스캔 자세를 깔고 IK 로 거른다 → data/{object}/scan/{name}/plan.npz.

계획 로직은 planning.plan_scan 에 있다. 이 스크립트는 ROI 를 정하고 저장만 한다.

물체 pose 를 주면(--object-position/--object-quat, Isaac 이 스테이지에서 읽어 넘긴다) 씬 배치
대신 그것을 쓰고 지지대도 거기 맞춘다. 계획에는 그 pose 와 장애물 OBB 를 함께 남긴다 — 복원·
Studio 가 촬영 당시의 셀로 점을 지우게 하려는 것이다(storage.ScanPlan).

관심 영역을 안 주면 시뮬레이션 편의상 정답 메시(source.obj)의 배치 bbox 에서 만든다
(sim.roi_from_ground_truth) — 도면 없는 상황에서는 사용자가 줘야 하는 값이다.

Usage:
    uv run scripts/core/reconstruction/scan_plan.py --object cylinder_sample
    uv run scripts/core/reconstruction/scan_plan.py --object sample \\
        --roi-center -0.73 -0.17 0.25 --roi-radius 0.15 --n-views 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config, scene_config  # noqa: E402
from core.reconstruction import storage  # noqa: E402
from core.reconstruction.storage import config_object_pose, snapshot_obstacles  # noqa: E402
from core.reconstruction.planning import plan_scan  # noqa: E402
from core.reconstruction.sensor import DEFAULT_SCAN_CAMERA  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True,
                   help="배치(씬 object_placements)와, ROI 기본값을 만들 정답 메시의 이름")
    scene_config.add_cli_argument(p)
    p.add_argument("--name", default="default", help="결과 폴더 data/{object}/scan/{name}")
    p.add_argument("--roi-center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="관심 영역 구의 중심 (world, m). 없으면 정답 메시 bbox 에서")
    p.add_argument("--roi-radius", type=float, help="관심 영역 구의 반경 (m)")
    p.add_argument("--standoff", type=float, default=None,
                   help="카메라 ↔ ROI 중심 거리 (m). 기본 = 센서 최적 거리")
    p.add_argument("--n-views", type=int, default=48, help="구면 위 후보 방향 수")
    p.add_argument("--min-elev", type=float, default=15.0, help="최저 앙각 (deg)")
    p.add_argument("--max-elev", type=float, default=85.0, help="최고 앙각 (deg)")
    p.add_argument("--ik-seeds", type=int, default=32)
    p.add_argument("--object-position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="물체 바닥 중심 (robot frame, m) — 씬 배치 대신 (Isaac 스테이지 pose)")
    p.add_argument("--object-quat", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"),
                   help="물체 회전 (w,x,y,z) — 씬 배치 대신")
    return p.parse_args()


def main():
    args = parse_args()
    scene_config.apply_cli(args, config)
    config.apply_object_placement(args.object)
    pose_source = "scene_placement"
    if args.object_position is not None:
        config.TARGET_OBJECT["position"] = np.asarray(args.object_position, dtype=np.float64)
        pose_source = "cli"
    if args.object_quat is not None:
        config.TARGET_OBJECT["rotation"] = np.asarray(args.object_quat, dtype=np.float64)
        pose_source = "cli"
    # 지지대는 물체 바닥에 맞춰 늘고 준다. planning.collision_world 는 config.OBSTACLES 를 그대로
    # 읽으므로(build_collision_world 처럼 다시 맞추지 않는다) 덮어쓴 뒤 여기서 맞춰야 한다.
    config.sync_support_to_target()

    if args.roi_center is not None and args.roi_radius is not None:
        center, radius = np.asarray(args.roi_center, dtype=np.float64), float(args.roi_radius)
        roi_source = "cli"
    else:
        from core.reconstruction.sim import roi_from_ground_truth

        center, radius = roi_from_ground_truth(args.object)
        roi_source = "ground_truth_bbox"
        print(f"[plan] ROI from ground-truth bbox (sim convenience): "
              f"center={np.round(center, 3).tolist()} r={radius:.3f} m")

    try:
        plan = plan_scan(DEFAULT_SCAN_CAMERA, center, radius, standoff=args.standoff,
                         n_views=args.n_views, min_elev_deg=args.min_elev,
                         max_elev_deg=args.max_elev, ik_seeds=args.ik_seeds)
    except ValueError as exc:
        sys.exit(f"[plan] {exc}")
    plan.meta = {"object": args.object, "scene": config.ACTIVE_SCENE, "roi_source": roi_source,
                 "object_pose_source": pose_source, **plan.meta}
    plan.object_pose = config_object_pose()
    plan.obstacles, plan.obstacle_names = snapshot_obstacles()

    out = storage.scan_dir(args.object, args.name)
    storage.save_plan(out, plan)
    elev = np.degrees(np.arcsin(plan.view_dirs[:, 2]))
    print(f"[plan] {len(plan.q)}/{plan.meta['n_candidates']} directions reachable "
          f"(elev {elev.min():.0f}–{elev.max():.0f}°) -> {out.relative_to(PROJECT_ROOT)}/plan.npz")


if __name__ == "__main__":
    main()

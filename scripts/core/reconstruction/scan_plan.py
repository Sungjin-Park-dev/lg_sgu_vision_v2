#!/usr/bin/env python3
"""관심 영역(구) 둘레에 스캔 자세를 깔고 IK 로 거른다 → data/{object}/scan/{name}/plan.npz.

계획 로직은 planning.plan_scan 에 있다(기본은 고리별 지그재그 경로, --layout pattern 은 고정
11장 패턴). 이 스크립트는 ROI·물체 방향을 정하고, 장 사이를 잇는 실행 궤적(scan_motion →
trajectory.csv)까지 계획해 저장한다. 궤적을 못 이으면 경고만 하고 계획은 남긴다(촬영은 순간이동).

물체 pose 를 주면(--object-position/--object-quat, Isaac 이 스테이지에서 읽어 넘긴다) 씬 배치
대신 그것을 쓰고 지지대도 거기 맞춘다. 계획에는 그 pose 와 장애물 OBB 를 함께 남긴다 — 복원·
Studio 가 촬영 당시의 셀로 점을 지우게 하려는 것이다(storage.ScanPlan).

관심 영역(ROI) 구: 중심을 안 주면 **물체 놓는 위치**(물체 pose = 바닥 중심)를 쓴다. 도면이 없어도
어디 놓였는지는 안다(ArUco 마커 등). 반경은 물체 크기를 대충 — 넉넉해도 결과가 같다(넓어진 ROI 로
들어온 받침·테이블 점은 씬 장애물로 지워진다). Isaac 패널의 "ROI at Placement" 와 같은 규칙이다.

Usage:
    uv run scripts/core/reconstruction/scan_plan.py --object cylinder_sample --roi-radius 0.12
    uv run scripts/core/reconstruction/scan_plan.py --object sample \\
        --roi-center -1.12 -0.25 0.07 --roi-radius 0.22 --all-views
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
from common.math_utils import quaternion_to_rotation_matrix  # noqa: E402
from core.reconstruction.planning import SCAN_PATTERN, path_stats, plan_scan  # noqa: E402
from core.reconstruction.scan_motion import MotionError, plan_scan_motion  # noqa: E402
from core.reconstruction.sensor import DEFAULT_SCAN_CAMERA  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True,
                   help="물체 이름 — 배치(씬 object_placements)와 결과 폴더를 정한다")
    scene_config.add_cli_argument(p)
    p.add_argument("--name", default="default", help="결과 폴더 data/{object}/scan/{name}")
    p.add_argument("--roi-center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="관심 영역 구의 중심 (world, m). 없으면 물체 놓는 위치(물체 pose)")
    p.add_argument("--roi-radius", type=float, default=0.20,
                   help="관심 영역 구의 반경 (m) — 물체 크기를 대충 (기본 0.20)")
    p.add_argument("--standoff", type=float, default=None,
                   help="카메라 ↔ ROI 중심 거리 (m). 기본 = 센서 최적 거리")
    p.add_argument("--layout", choices=("zigzag", "pattern", "all"), default="zigzag",
                   help="어느 방향에서 찍나 — zigzag(기본): 고리별 지그재그 경로, pattern: 고정 11장, "
                        "all: 닿는 후보 전부(실험용)")
    p.add_argument("--all-views", action="store_true", help="--layout all 과 같다")
    p.add_argument("--zigzag-rings", type=int, default=2, help="zigzag: 정수리 아래 고리 수")
    p.add_argument("--zigzag-spacing", type=float, default=40.0,
                   help="zigzag: 고리 위 장 사이 간격 (deg, 구면 호 기준)")
    p.add_argument("--n-views", type=int, default=48,
                   help="위쪽 띠 후보 방향 수 — 못 닿는 패턴 방향을 대신할 풀 (--all-views 면 찍을 후보)")
    p.add_argument("--min-elev", type=float, default=15.0, help="최저 앙각 (deg)")
    p.add_argument("--max-elev", type=float, default=85.0, help="최고 앙각 (deg)")
    p.add_argument("--ik-seeds", type=int, default=32)
    p.add_argument("--start-joints", type=float, nargs=6, default=None, metavar="Q",
                   help="로봇의 지금 관절값 (rad) — 궤적의 시작·끝. 기본 ROBOT_START_STATE "
                        "(Isaac 패널은 현재 자세를 넘긴다)")
    p.add_argument("--no-motion", action="store_true",
                   help="장 사이 실행 궤적(trajectory.csv)을 계획하지 않는다")
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

    if args.roi_center is not None:
        center, roi_source = np.asarray(args.roi_center, dtype=np.float64), "cli"
    else:
        center, roi_source = np.asarray(config.TARGET_OBJECT["position"], dtype=np.float64), "placement"
    radius = float(args.roi_radius)
    print(f"[plan] ROI ({roi_source}): center {np.round(center, 4).tolist()} r {radius:.3f} m")
    # 고정 패턴은 물체 방향을 따라 돈다 — 물체 x축의 수평 방위.
    R = quaternion_to_rotation_matrix(np.asarray(config.TARGET_OBJECT["rotation"], dtype=np.float64))
    yaw_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    start = (np.asarray(args.start_joints, dtype=np.float64) if args.start_joints is not None
             else np.asarray(config.ROBOT_START_STATE, dtype=np.float64))
    try:
        plan = plan_scan(DEFAULT_SCAN_CAMERA, center, radius, standoff=args.standoff,
                         n_views=args.n_views, min_elev_deg=args.min_elev,
                         max_elev_deg=args.max_elev, ik_seeds=args.ik_seeds,
                         layout="all" if args.all_views else args.layout, pattern=SCAN_PATTERN,
                         yaw_deg=yaw_deg, zigzag_rings=args.zigzag_rings,
                         zigzag_spacing_deg=args.zigzag_spacing, start=start)
    except ValueError as exc:
        sys.exit(f"[plan] {exc}")
    plan.meta = {"object": args.object, "scene": config.ACTIVE_SCENE, "robot": config.ROBOT["name"],
                 "roi_source": roi_source,
                 "object_pose_source": pose_source, **plan.meta}
    plan.object_pose = config_object_pose()
    plan.obstacles, plan.obstacle_names = snapshot_obstacles()

    out = storage.scan_dir(args.object, args.name)
    storage.clear_motion(out)                 # 옛 계획의 궤적이 새 계획에 붙어 쓰이면 안 된다
    if not args.no_motion:
        try:
            plan.meta["motion"], q = plan_scan_motion(plan, start, out)
            if not np.array_equal(q, plan.q):       # 막힌 이동을 2π 동치로 고쳤다 — 자세(FK)는 같다
                plan.q = q
                plan.meta["path"] = path_stats(q, start)
        except MotionError as exc:
            print(f"[motion] ⚠️  {exc} — no trajectory; capture will teleport between poses")
            plan.meta["motion"] = {"error": str(exc)}
    storage.save_plan(out, plan)
    elev = np.degrees(np.arcsin(plan.view_dirs[:, 2]))
    print(f"[plan] {len(plan.q)} shots (elev {elev.min():.0f}–{elev.max():.0f}°) "
          f"-> {out.relative_to(PROJECT_ROOT)}/plan.npz")


if __name__ == "__main__":
    main()

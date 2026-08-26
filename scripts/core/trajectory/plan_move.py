#!/usr/bin/env python3
"""임의의 두 관절자세 사이를 충돌-free 로 잇는 궤적 하나를 계획해 CSV(+npz)로 낸다.

저장소에 "시작 q + 목표 q → 계획된 궤적" 진입점이 없어서 만든 것이다. 스캔 궤적을 만드는
경로(`glns/solve.py` → `glns/verify.py`)는 viewpoint 에서 출발하고, `glns/verify.py
--home-transitions-only` 는 A/B 가 (HOME, scan[0]) / (scan[-1], HOME) 로 고정돼 있다.
그래서 "지금 로봇이 있는 자리에서 저기까지" 를 계획할 방법이 없었다 — isaac_pipeline 의
HOME↔스캔시작 버튼이 계획 없이 직선으로 움직이던 이유다.

새 알고리즘은 없다. 기존 조각의 조립이다:
    build_collision_world → build_reconfig_motion_planner → plan_reconfig_transits
    → interpolate_and_resample → collision_gate_and_save
계획 사다리(direct → via-roll → via-tilt → via-home)를 그대로 물려받는다.

물체 pose 는 `--object-position/--object-quat` 로 받아 `config.TARGET_OBJECT` 에 덮어쓴다 —
`glns/solve.py` 가 쓰는 것과 같은 관례다(호출자가 살아있는
기즈모 pose 를 넘긴다).

Exit: 0 = 충돌-free CSV 생성, 2 = 경로 없음 / 충돌 게이트 실패 / 인자 오류(argparse).
시작과 목표가 같으면 실패가 아니다 — 이동 없는 2행 CSV 를 내고 0 을 반환한다.

Usage:
    uv run --no-sync python scripts/core/trajectory/plan_move.py \\
        --object sample --object-position -0.15 0.741 0.19 \\
        --object-quat 0.70710678 0 0 0.70710678 \\
        --from-joints "1.5,-1.5,2.0,-0.5,1.5,0.0" \\
        --to-joints   "0.9,-1.2,1.7,-0.4,1.4,0.1" \\
        --output data/sample/trajectory/74/home_move_return.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config, scene_config  # noqa: E402
from core import trajectory as PT  # noqa: E402
from core.glns.candidates import _joint_limits_and_periods  # noqa: E402

# 엣지가 하나뿐이라 큰 배치는 낭비다. warmup 버퍼가 배치 슬롯당 ~375MB 라 기본값(24)이면
# ~9GB 를 잡는데, 이 CLI 는 Isaac Sim 이 이미 GPU 를 쓰고 있는 동안 호출된다.
MOVE_PLANNER_BATCH = 1


def _joints(text: str) -> np.ndarray:
    """argparse ``type=`` — "q0,...,q5" → (6,) ndarray.

    argparse 가 옵션 이름을 붙여 rc=2 로 죽여주므로 여기서 직접 SystemExit 하지 않는다.
    """
    values = [v for v in text.replace(",", " ").split() if v]
    try:
        q = np.array([float(v) for v in values], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"could not parse as numbers ({exc})")
    if q.shape != (6,):
        raise argparse.ArgumentTypeError(f"expected 6 joint values (got {len(values)})")
    return q


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 첫 값이 음수면 '-1.5,...' 로 시작해 argparse 가 옵션으로 오인한다 — 호출자는
    # --from-joints=<v> 형태(공백 아님)로 넘길 것.
    p.add_argument("--from-joints", required=True, type=_joints,
                   help='시작 관절값 "q0,...,q5" [rad]')
    p.add_argument("--to-joints", required=True, type=_joints,
                   help='목표 관절값 "q0,...,q5" [rad]')
    p.add_argument("--object", required=True, help="충돌 세계에 놓을 물체 이름")
    scene_config.add_cli_argument(p)
    p.add_argument("--object-position", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   default=None, help="물체 위치 (robot base_link frame, m)")
    p.add_argument("--object-quat", type=float, nargs=4, metavar=("W", "X", "Y", "Z"),
                   default=None, help="물체 회전 쿼터니언")
    p.add_argument("--output", required=True, type=Path, help="출력 CSV 경로")
    p.add_argument("--spacing", type=float, default=PT.DEFAULT_SPACING_M,
                   help=f"resample 간격 (기본 {PT.DEFAULT_SPACING_M})")
    p.add_argument("--wd-m", type=float, default=None,
                   help="working distance [m]. 주면 사다리에 via-tilt 단계가 추가된다")
    p.add_argument("--no-via", action="store_true",
                   help="via 사다리(roll/tilt/home) 없이 direct 만 시도")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    q_from, q_to = args.from_joints, args.to_joints

    print("=" * 60)
    print("PLAN MOVE (collision-free joint-to-joint)")
    print("=" * 60)
    print(f"  object : {args.object}")
    print(f"  from   : {np.round(np.rad2deg(q_from), 1).tolist()} deg")
    print(f"  to     : {np.round(np.rad2deg(q_to), 1).tolist()} deg")

    # 물체 배치: 기본값 → CLI override. build_collision_world 가 호출 시점에 읽으므로
    # 여기서 config 를 덮어쓰면 그대로 반영된다(pipeline.py / solve.py 와 같은 관례).
    # 씬을 먼저 반영한다 — 물체 배치(object_placements)가 이제 씬 소유다.
    scene_config.apply_cli(args, config)
    config.apply_object_placement(args.object)
    if args.object_position is not None:
        config.TARGET_OBJECT["position"] = np.array(args.object_position, dtype=np.float64)
    if args.object_quat is not None:
        config.TARGET_OBJECT["rotation"] = np.array(args.object_quat, dtype=np.float64)
    print(f"  placement pos={np.round(config.TARGET_OBJECT['position'], 3).tolist()} "
          f"quat={np.round(config.TARGET_OBJECT['rotation'], 4).tolist()}")

    robot_cfg = PT.resolve_robot_config(PT.ROBOT_CONFIG)
    world_config = PT.build_collision_world(args.object)

    # 목표를 **시작 자세에 가장 가까운 2π 등가로 옮긴다.** UR20 은 6축 중 5축이 ±354° 라
    # 같은 자세를 여러 관절값으로 쓸 수 있는데, 그걸 안 맞추면 같은 곳으로 가면서도 한 바퀴를
    # 돈다(실측: pan 이 +2.82 rad 에서 -2.69 rad 로 315.9° 회전 — 등가인 +3.59 rad 로 가면
    # 44.1°). 자세는 그대로이므로 도달 결과는 바뀌지 않고 이동량만 준다.
    # elbow(±174°)는 등가가 한계 밖이라 그대로 남는다 — 물리적으로 짧게 갈 방법이 없다.
    joint_lower, joint_upper, joint_periods = _joint_limits_and_periods(robot_cfg)
    q_to_aligned = PT.align_to_reference(
        q_to, q_from, joint_periods, joint_lower, joint_upper)
    if not np.allclose(q_to_aligned, q_to):
        moved = np.rad2deg(np.abs(q_to - q_from)).max()
        after = np.rad2deg(np.abs(q_to_aligned - q_from)).max()
        print(f"  aligned target to the start pose: max joint travel "
              f"{moved:.1f} -> {after:.1f} deg")
        print(f"    to (aligned): {np.round(np.rad2deg(q_to_aligned), 1).tolist()} deg")
        q_to = q_to_aligned

    # 엣지 1개짜리 batch — plan_seams_batched(core/glns/joining.py)의 1-pair 형태다.
    selected = np.stack([q_from, q_to])

    if np.max(np.abs(q_to - q_from)) < 1e-6:
        # 이미 목표 자세다. 실패가 아니므로 출력 계약(항상 CSV)을 지킨다 — 호출자가
        # "경로 없음" 과 "이동 없음" 을 분기할 필요가 없다. 플래너는 건너뛰지만 아래
        # 충돌 게이트는 그대로 지난다: 지금 자세가 이미 충돌 중이면 정직하게 실패한다.
        print("  already at the goal pose - emitting a 2-row, no-motion trajectory.")
        traj = selected
    else:
        planner = PT.build_reconfig_motion_planner(
            robot_cfg, world_config, batch_size=MOVE_PLANNER_BATCH)
        segments, stats = PT.plan_reconfig_transits(
            selected, np.array([0], dtype=np.int64), robot_cfg, world_config,
            wd_m=args.wd_m, enable_via_ladder=not args.no_via,
            lock_wrist3=False, motion_planner=planner,
        )
        waypoints = segments.get(0)
        route = next((s.get("route") for s in stats if s.get("success")), None)
        if waypoints is None:
            print("  x no collision-free route found (direct and via ladder both failed).")
            return 2
        print(f"  OK planned via '{route}' - {len(waypoints)} raw waypoints")

        # 이동 전체가 transit 이므로 마스크는 all-True. (2행 입력의 첫 노드를 scan 으로
        # 타이핑하는 interpolate_and_resample 의 기본 동작을 덮는다 — resample_seam 과 동일.)
        traj, _is_transit, _dropped, _runs = PT.interpolate_and_resample(
            selected, {0: waypoints}, robot_cfg,
            mode=PT.RESAMPLE_MODE, spacing=args.spacing,
            reconfig_threshold_rad=np.deg2rad(PT.RECONFIG_THRESHOLD_DEG),
            world_scene=world_config,
        )
    is_transit = np.ones(len(traj), dtype=bool)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gate = PT.collision_gate_and_save(
        traj, is_transit, robot_cfg=robot_cfg, world_config=world_config,
        out_csv=args.output,
    )
    if not gate["collision_free"]:
        print(f"  x collision gate failed: {gate['n_collisions']} collisions - not saved.")
        return 2

    # 끝점은 계획의 전제다 — 실행기가 CSV 첫 행과 현재 자세가 다르면 계획되지 않은 직선을
    # 앞에 덧붙이므로, 첫 행이 정말 q_from 이어야 이 궤적이 '전 구간 계획됨' 이다.
    d0 = float(np.max(np.abs(traj[0] - q_from)))
    d1 = float(np.max(np.abs(traj[-1] - q_to)))
    print(f"  endpoints: |traj[0]-from|={d0:.2e} rad, |traj[-1]-to|={d1:.2e} rad")
    print(f"  waypoints={gate['n_waypoints']}, duration={gate['total_time']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

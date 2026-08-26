"""계획된 궤적의 최종 충돌 게이트 + 저장.

`plan_cspace` 의 success 는 신뢰하지 않는다 — densify 해서 실제 침투를 다시 확인하고,
충돌-free 일 때만 FK/시간을 계산해 CSV(+npz sidecar)를 쓴다. 실행용 산출물이 나가는
마지막 관문이라 계획 경로가 무엇이든(GLNS 성분·joined·HOME 이동) 여기를 지난다.

``out_csv=None`` 이면 검사와 시간 계산만 하고 파일은 쓰지 않는다 — 중간 산출물을 디스크에
남기지 않으면서 성분별 리포트는 그대로 뽑기 위한 것이다(verify.py 의 성분 표).

`motion` 위 계층에 둔다: motion/robot/timing/storage 를 모두 쓰므로 그 안에 넣으면 순환이 된다.
"""

from __future__ import annotations

import json

import numpy as np

from . import settings
from .motion import densify_for_collision_check
from .robot import batch_collision_check, compute_fk
from .storage import save_trajectory_csv
from .timing import compute_trajectory_times


def collision_gate_and_save(final_traj, final_is_transit, *, robot_cfg, world_config,
                            out_csv=None, meta=None, kinds=None):
    """densify 충돌게이트 → 충돌-free 면 FK/시간 계산, ``out_csv`` 가 있으면 CSV/npz 저장.

    Args:
        out_csv: None 이면 저장하지 않는다(검사·시간만). 반환 dict 의 ``csv`` 도 None.
        kinds: 행별 출처 라벨(viewpoint/interpolated/planned). 주면 CSV 컬럼과 npz 에 함께
            남긴다 — 어떤 점인지는 궤적을 다시 만들지 않으면 복원할 수 없는 정보다.
        meta: npz 에 함께 남길 생성 파라미터. CSV 스키마에는 넣을 자리가 없어서 sidecar 로
            간다 — 예전에 DP 가 파일명에 인코딩하던 정보의 자리다. 속도 노브는 호출자가
            챙길 필요 없이 ``meta["timing"]`` 으로 여기서 자동으로 들어간다.

    속도 노브는 ``settings`` 를 **호출 시점에** 읽는다. CLI/UI 가 apply_timing_cli() 로
    덮어쓴 값이 그대로 걸리게 하려는 것이다(모듈 최상단에서 값을 import 해 오면 override 가
    조용히 무시된다).
    """
    collision_traj = densify_for_collision_check(final_traj)
    _, n_collisions = batch_collision_check(collision_traj, robot_cfg, world_config)
    collision_free = n_collisions == 0

    total_time = transit_time = float("nan")
    if collision_free:
        ee_positions, ee_quaternions = compute_fk(final_traj, robot_cfg)
        traj_times, time_stats = compute_trajectory_times(
            final_traj, ee_positions, ee_quaternions,
            ee_speed_m_s=settings.EE_SPEED_MM_S / 1000.0,
            ee_angular_speed_rad_s=np.deg2rad(settings.EE_ANGULAR_SPEED_DEG_S),
            max_joint_vel_rad_s=settings.MAX_JOINT_VEL_RAD_S,
            min_segment_dt=settings.MIN_SEGMENT_DT_S,
            corner_angle_threshold_rad=np.deg2rad(settings.CORNER_ANGLE_THRESHOLD_DEG),
            corner_max_slowdown=settings.CORNER_MAX_SLOWDOWN,
            is_transit=final_is_transit,
        )
        total_time = float(time_stats["total_time"])
        transit_time = float(time_stats["transit_time"])
        if out_csv is not None:
            save_trajectory_csv(
                final_traj, ee_positions, ee_quaternions, str(out_csv), times=traj_times,
                kinds=kinds,
            )
            # trajectory_studio.py 의 dense 재생용 sidecar: CSV 에 없는 is_transit 마스크와
            # FK EE 위치를 함께 저장(실행용 CSV 스키마는 건드리지 않는다).
            np.savez(
                out_csv.with_suffix(".npz"),
                joints=np.asarray(final_traj, dtype=np.float64),
                ee_positions=np.asarray(ee_positions, dtype=np.float64),
                is_transit=np.asarray(final_is_transit, dtype=bool),
                waypoint_kind=(np.zeros(len(final_traj), dtype=np.int8) if kinds is None
                               else np.asarray(kinds, dtype=np.int8)),
                times=np.asarray(traj_times, dtype=np.float64),
                # 속도 노브는 여기서 넣는다 — 호출자마다 챙기면 하나쯤 빠지고, 그러면 그
                # 궤적이 어떤 속도로 구워졌는지 되짚을 방법이 없다.
                meta=json.dumps({**(meta or {}), "timing": settings.timing_values()},
                                ensure_ascii=False),
            )

    saved = collision_free and out_csv is not None
    return {
        "n_collisions": int(n_collisions),
        "collision_free": collision_free,
        "total_time": total_time,
        "transit_time": transit_time,
        "n_waypoints": len(final_traj),
        "csv": str(out_csv) if saved else None,
    }

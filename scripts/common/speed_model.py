"""Motion Speed 의 결합 모델 — 배율 둘이 노브 넷을 정한다.

구간의 종류마다 다른 쌍이 시간을 정하므로 배율도 둘이다:

  스캔 배율 f_s (보간 구간 = viewpoint 사이, tilt 부채꼴):
      EE 선속도 = f_s × 10 mm/s,   EE 각속도 = f_s × 20 deg/s
  이동 배율 f_p (플래닝 구간 = transit, Plan to Start/HOME):
      관절 속도 배율 = f_p × vs0,   관절 가속도 배율 = f_p² × as0
      (vs0, as0 = moveit/ur_config/joint_limits.yaml 의 default_*_scaling_factor — MoveIt 과 같은 수)

속도 ∝ f, 가속도 ∝ f² 인 이유: 궤적을 시간축으로 1/f 압축하면 정확히 그렇게 된다. 그래서 배율은
곧 "그 구간의 시간 배율" 이다 — 이동 배율 2 는 같은 모양의 이동을 절반 시간에 한다.

Isaac 앱(패널)과 core/trajectory/settings.py(기본값) 가 둘 다 여기서 읽는다. numpy·yaml 만 쓴다.
"""

from __future__ import annotations

import math

from . import joint_limits as _jl

# 스캔 쪽 기준값(f_s = 1). settings.EE_SPEED_MM_S / EE_ANGULAR_SPEED_DEG_S 의 출처.
SCAN_BASE = {"ee_mm_s": 10.0, "ee_angular_deg_s": 20.0}
SCAN_FACTOR_MIN, SCAN_FACTOR_MAX = 0.1, 3.0

# 이동 쪽 배율 범위. 상한은 두 scaling factor 가 1.0(로봇 한계 100%)을 넘지 않는 최대 f_p —
# MoveIt 도 1.0 을 넘는 배율을 받지 않는다.
TRANSIT_FACTOR_MIN = 0.1


def transit_base() -> tuple[float, float]:
    """(vs0, as0): f_p = 1 에서의 관절 속도·가속도 배율. joint_limits.yaml 이 정한다."""
    lim = _jl.load_joint_limits()
    return lim.default_velocity_scaling, lim.default_acceleration_scaling


def transit_factor_max() -> float:
    vs0, as0 = transit_base()
    return math.floor(min(1.0 / vs0, math.sqrt(1.0 / as0)) * 100.0) / 100.0


def scan_values(f: float) -> dict:
    return {"ee_mm_s": f * SCAN_BASE["ee_mm_s"],
            "ee_angular_deg_s": f * SCAN_BASE["ee_angular_deg_s"]}


def transit_values(f: float) -> dict:
    vs0, as0 = transit_base()
    return {"velocity_scaling": f * vs0, "acceleration_scaling": f * f * as0}


def scan_factor_from(key: str, value: float) -> float:
    """스캔 쪽 숫자 하나에서 f_s 를 역산한다."""
    return float(value) / SCAN_BASE[key]


def transit_factor_from(key: str, value: float) -> float:
    """이동 쪽 숫자 하나에서 f_p 를 역산한다(가속도는 f² 이라 제곱근)."""
    vs0, as0 = transit_base()
    if key == "velocity_scaling":
        return float(value) / vs0
    if key == "acceleration_scaling":
        return math.sqrt(max(0.0, float(value)) / as0)
    raise KeyError(key)


def clamp_scan(f: float) -> float:
    return min(SCAN_FACTOR_MAX, max(SCAN_FACTOR_MIN, float(f)))


def clamp_transit(f: float) -> float:
    return min(transit_factor_max(), max(TRANSIT_FACTOR_MIN, float(f)))


def cli_args(f_scan: float, f_transit: float) -> str:
    """두 배율 → settings.add_timing_cli_arguments 가 받는 플래그 문자열(범위로 자른 뒤)."""
    s = scan_values(clamp_scan(f_scan))
    t = transit_values(clamp_transit(f_transit))
    return (f"--ee-speed-mm-s {s['ee_mm_s']:g} "
            f"--ee-angular-speed-deg-s {s['ee_angular_deg_s']:g} "
            f"--joint-velocity-scaling {t['velocity_scaling']:.4g} "
            f"--joint-acceleration-scaling {t['acceleration_scaling']:.4g}")

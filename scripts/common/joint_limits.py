"""MoveIt 과 공유하는 관절 한계 파일의 numpy 전용 로더.

파일은 ``scripts/moveit/ur_config/joint_limits.yaml`` 하나다. move_group 은 launch 가 그 파일을
``robot_description_planning`` 으로 넘겨 읽고, Inspection 의 시간매김(core/trajectory/timing.py)과
Isaac 앱의 Motion Speed 패널은 여기서 읽는다. 세 곳이 같은 수를 보게 하려는 것이므로 한계나
기본 배율을 바꾸려면 **yaml 을 고친다** — 여기에 값을 적지 않는다.

cuRobo 를 끌어오지 않는다(numpy + yaml). Isaac 프로세스에서도 import 할 수 있어야 해서다.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

JOINT_LIMITS_YAML = Path(__file__).resolve().parents[1] / "moveit" / "ur_config" / "joint_limits.yaml"

# 궤적 배열의 열 순서. cuRobo 로봇 yml 의 cspace.joint_names, UR 드라이버의 관절 순서와 같다.
UR_JOINT_ORDER = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


@dataclass(frozen=True)
class JointLimits:
    """관절별 한계(로봇 100%)와 파일이 정한 기본 배율."""

    names: tuple
    max_velocity: np.ndarray        # (dof,) rad/s
    max_acceleration: np.ndarray    # (dof,) rad/s²
    default_velocity_scaling: float
    default_acceleration_scaling: float

    def ordered(self, names) -> "JointLimits":
        """다른 관절 순서로 재배열한 사본. 궤적 열 순서가 파일 순서와 다를 때 쓴다."""
        names = tuple(names)
        idx = [self.names.index(n) for n in names]      # 없는 이름이면 ValueError — 조용히 넘기지 않는다
        return JointLimits(names, self.max_velocity[idx], self.max_acceleration[idx],
                           self.default_velocity_scaling, self.default_acceleration_scaling)


def _scaling(value, key: str) -> float:
    v = float(value)
    if not 0.0 < v <= 1.0:
        raise ValueError(f"{key} must be in (0, 1], got {v} ({JOINT_LIMITS_YAML})")
    return v


@functools.lru_cache(maxsize=None)
def load_joint_limits(path: Path = JOINT_LIMITS_YAML, names: tuple = UR_JOINT_ORDER) -> JointLimits:
    """yaml → JointLimits. 관절마다 속도·가속도 한계가 둘 다 있어야 한다(없으면 ValueError).

    MoveIt 은 없는 한계를 "무제한" 으로 두지만 우리 시간매김은 나눗셈에 쓰므로 빠진 값을
    허용하지 않는다. 한 번 읽어 캐시한다(파일은 리포 자산이라 실행 중 바뀌지 않는다).
    """
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    per_joint = doc.get("joint_limits") or {}
    vel, acc = [], []
    for name in names:
        entry = per_joint.get(name)
        if entry is None:
            raise ValueError(f"{path}: no joint_limits entry for '{name}'")
        if not entry.get("has_velocity_limits") or float(entry.get("max_velocity", 0.0)) <= 0.0:
            raise ValueError(f"{path}: '{name}' needs has_velocity_limits + max_velocity > 0")
        if not entry.get("has_acceleration_limits") or float(entry.get("max_acceleration", 0.0)) <= 0.0:
            raise ValueError(f"{path}: '{name}' needs has_acceleration_limits + max_acceleration > 0")
        vel.append(float(entry["max_velocity"]))
        acc.append(float(entry["max_acceleration"]))
    limits = JointLimits(
        tuple(names), np.asarray(vel, dtype=np.float64), np.asarray(acc, dtype=np.float64),
        _scaling(doc.get("default_velocity_scaling_factor", 0.1), "default_velocity_scaling_factor"),
        _scaling(doc.get("default_acceleration_scaling_factor", 0.1), "default_acceleration_scaling_factor"),
    )
    limits.max_velocity.flags.writeable = False
    limits.max_acceleration.flags.writeable = False
    return limits

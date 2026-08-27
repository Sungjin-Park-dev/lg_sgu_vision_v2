"""궤적 CSV(+npz sidecar)를 읽는 numpy-only 로더.

`core/trajectory/` 가 아니라 여기 사는 이유는 `common/tilt_geometry.py` 와 같다:
Isaac 앱(isaac_pipeline.py)이 이 함수들을 쓰는데, `core.trajectory` 를 import 하면
패키지 __init__ 이 cuRobo 와 torch 를 통째로 끌어온다. Isaac 프로세스에서는 그게
(a) 첫 호출에 UI 를 몇 초 멈추고 (b) 생성 서브프로세스가 GPU 를 쓰는 동안 CUDA
컨텍스트를 새로 잡는다. 여기 필요한 건 CSV 열 몇 개와 JSON 하나뿐이다.

쓰는 곳: core/trajectory/tilt_motion.py (CLI), apps/isaac_pipeline.py (UI).
둘이 같은 함수를 봐야 화면의 부채꼴과 실제로 생성될 포즈가 어긋나지 않는다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .math_utils import quaternion_to_rotation_matrix

# CSV 가 들고 있는 카메라 포즈 열. 회전은 (x, y, z, w) 순이다.
CAMERA_POSE_COLUMNS = (
    "target-POS_X", "target-POS_Y", "target-POS_Z",
    "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W",
)


def load_trajectory_row(csv_path: Path, row_index: int):
    """궤적 CSV 의 한 행 → (카메라 4x4 포즈, 라벨, 총 행 수).

    CSV 는 이미 각 행의 카메라 포즈를 갖고 있다(target-POS_* / target-ROT_*, FK 산출).
    그래서 viewpoint h5 를 다시 읽어 자세를 재구성할 필요가 없고, **보간·모션플래닝으로
    생긴 행도 중심으로 삼을 수 있다** — viewpoint h5 에는 그런 점이 아예 없다.

    라벨(waypoint_kind)은 있으면 읽고 없으면 "unknown". 옛 CSV 에는 그 컬럼이 없다.
    """
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"trajectory CSV has no rows: {csv_path}")
    if not 0 <= row_index < len(rows):
        raise IndexError(f"row {row_index} out of range (CSV has {len(rows)} rows)")
    row = rows[row_index]
    need = CAMERA_POSE_COLUMNS
    missing = [c for c in need if c not in row]
    if missing:
        raise ValueError(f"trajectory CSV lacks camera pose columns: {missing}")
    pos = np.array([float(row[c]) for c in need[:3]], dtype=np.float64)
    qx, qy, qz, qw = (float(row[c]) for c in need[3:])
    pose = np.eye(4, dtype=np.float64)
    # CSV 는 (x,y,z,w), quaternion_to_rotation_matrix 는 (w,x,y,z) 를 받는다.
    pose[:3, :3] = quaternion_to_rotation_matrix(np.array([qw, qx, qy, qz]))
    pose[:3, 3] = pos
    return pose, (row.get("waypoint_kind") or "unknown"), len(rows)


def load_trajectory_meta(csv_path: Path) -> dict:
    """CSV 옆 npz sidecar 의 meta. 작업거리와 물체 배치가 거기 박혀 있다."""
    npz = Path(csv_path).with_suffix(".npz")
    if not npz.exists():
        return {}
    try:
        with np.load(npz, allow_pickle=False) as d:
            return json.loads(str(d["meta"])) if "meta" in d else {}
    except (ValueError, TypeError, KeyError):
        return {}

# SPDX-License-Identifier: Apache-2.0
"""
Robot-frame 기준 environment + workcell 시각 확인용 standalone 스크립트.

규약: robot base = world origin. environment.usd 만 +X 방향으로 밀어
robot이 환경 내 적절한 자리에 위치하도록 정렬.

Usage:
    uv run scripts/isaac/load_environment.py
"""

import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UR20_DIR = PROJECT_ROOT / "ur20_description"

ENV_USD    = UR20_DIR / "environment.usd"
ROBOT_USD  = UR20_DIR / "ur20/ur20.usd"
MOUNT_USD  = UR20_DIR / "ur10_mount.usd"
TABLE_USD  = UR20_DIR / "thor_table.usd"
OBJECT_USD = PROJECT_ROOT / "data" / "sample" / "mesh" / "source.usd"

ROBOT_PATH  = "/World/UR20"
MOUNT_PATH  = "/World/Mount"
TABLE_PATH  = "/World/Table"
ENV_PATH    = "/World/Environment"
OBJECT_PATH = "/World/TargetObject"

for p in (ENV_USD, ROBOT_USD, MOUNT_USD, TABLE_USD, OBJECT_USD):
    if not p.exists():
        sys.exit(f"USD not found: {p}")

simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": False})

from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import prims, viewports
from pxr import Usd, UsdGeom


def print_bbox(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"  [bbox] {prim_path}: <invalid prim>")
        return
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    size = mx - mn
    print(
        f"  [bbox] {prim_path}\n"
        f"         min=({mn[0]:+.3f}, {mn[1]:+.3f}, {mn[2]:+.3f})  "
        f"max=({mx[0]:+.3f}, {mx[1]:+.3f}, {mx[2]:+.3f})  "
        f"size=({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})"
    )


simulation_context = SimulationContext(stage_units_in_meters=1.0)
viewports.set_camera_view(eye=np.array([2.5, 2.5, 1.8]), target=np.array([0, 0.5, 0.5]))

# --- 치수 / 위치 파라미터 -----------------------------------------------------
MOUNT_HEIGHT = 0.805    # 로봇 베이스 높이 (m)
TABLE_HEIGHT = 0.630    # 테이블 높이 (m)
MOUNT_USD_INTRINSIC_Z = 0.515  # bbox 측정 — Z 스케일 보정용
TABLE_USD_INTRINSIC_Z = 0.795

# Table USD 로컬 bbox center (X/Y) — robot frame에 정렬할 때 빼줌
TABLE_USD_BBOX_CENTER_X = 0.270
TABLE_USD_BBOX_CENTER_Y = -0.002

# Table center 목표 위치 (robot frame, meters) — 기존 config.TABLE 위치 승계
TABLE_TARGET_X = -0.2
TABLE_TARGET_Y = 1.1

# Mount XY 두께 배율 (Z는 height/intrinsic 비율 유지)
MOUNT_XY_SCALE = 2.0

# Target object (참고용 상수 — 실제 USD 로드는 추후 추가)
# 기존 config.TARGET_OBJECT.position = (-0.1, 1.1, 0.095) [old robot-base frame]
# old table top z = -0.07 → object center가 table top + 0.165m
# 새 table top = 0.630 → 새 object z = 0.630 + 0.165 = 0.795
TARGET_OBJECT_POSITION = np.array([-0.1, 1.1, 0.795])
TARGET_OBJECT_ROTATION = np.array([0.7071, 0.0, 0.0, 0.7071])  # w, x, y, z

# Environment USD 위치 — robot이 환경 내 어디에 서 있을지를 결정
ENV_OFFSET = np.array([2.0, 0.0, 0.0])

# --- 1) 월드 환경 ------------------------------------------------------------
prims.create_prim(
    ENV_PATH,
    "Xform",
    position=ENV_OFFSET,
    usd_path=str(ENV_USD),
)

# --- 2) Mount (origin이 상단) ------------------------------------------------
prims.create_prim(
    MOUNT_PATH,
    "Xform",
    position=np.array([0.0, 0.0, MOUNT_HEIGHT]),
    scale=np.array([MOUNT_XY_SCALE, MOUNT_XY_SCALE, MOUNT_HEIGHT / MOUNT_USD_INTRINSIC_Z]),
    usd_path=str(MOUNT_USD),
)

# --- 3) Robot (origin이 베이스, mount 위에 안착) -----------------------------
prims.create_prim(
    ROBOT_PATH,
    "Xform",
    position=np.array([0.0, 0.0, MOUNT_HEIGHT]),
    usd_path=str(ROBOT_USD),
)

# --- 4) Table (origin이 상단 + X/Y center 보정) ------------------------------
prims.create_prim(
    TABLE_PATH,
    "Xform",
    position=np.array([
        TABLE_TARGET_X - TABLE_USD_BBOX_CENTER_X,
        TABLE_TARGET_Y - TABLE_USD_BBOX_CENTER_Y,
        TABLE_HEIGHT,
    ]),
    scale=np.array([1.0, 1.0, TABLE_HEIGHT / TABLE_USD_INTRINSIC_Z]),
    usd_path=str(TABLE_USD),
)

# --- 5) Target object --------------------------------------------------------
# source.usd 내부에 scale=0.01 이 박혀 있어 명시적으로 (1,1,1) 로 덮어씀
prims.create_prim(
    OBJECT_PATH,
    "Xform",
    position=TARGET_OBJECT_POSITION,
    orientation=TARGET_OBJECT_ROTATION,
    scale=np.array([1.0, 1.0, 1.0]),
    usd_path=str(OBJECT_USD),
)

simulation_app.update()
simulation_context.initialize_physics()
simulation_context.play()

import omni.usd
_stage = omni.usd.get_context().get_stage()
print(f"[load_environment] Robot frame == world origin")
print(f"[load_environment] Environment offset: {ENV_OFFSET.tolist()}")
print("[load_environment] Bounding boxes (world coords, meters):")
for path in (ROBOT_PATH, MOUNT_PATH, TABLE_PATH, OBJECT_PATH, ENV_PATH):
    print_bbox(_stage, path)

while simulation_app.is_running():
    simulation_context.step(render=True)

simulation_context.stop()
simulation_app.close()

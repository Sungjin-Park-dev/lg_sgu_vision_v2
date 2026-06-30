#!/usr/bin/env python3
"""Compare the planner camera frame against the USD camera mount.

The motion planner uses the URDF/YAML end-effector link
``camera_optical_frame``. Isaac Sim publishes images from an ``InspectionCamera``
that is created under the USD ``camera_mount`` prim. This script compares both
transforms relative to the same parent frame, usually ``flange``.

Usage:
    uv run scripts/isaac/usd/check_camera_alignment.py
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USD = PROJECT_ROOT / "workcell" / "robot" / "ur20_with_camera.usd"
DEFAULT_URDF = PROJECT_ROOT / "workcell" / "robot" / "ur20_with_camera.urdf"
DEFAULT_USD_PARENT = "/Root/UR20/wrist_3_link/flange"
DEFAULT_USD_CAMERA = "/Root/UR20/wrist_3_link/flange/camera_mount/camera_optical_frame"


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def origin_transform(origin) -> np.ndarray:
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin is not None:
        if origin.get("xyz"):
            xyz = [float(v) for v in origin.get("xyz").split()]
        if origin.get("rpy"):
            rpy = [float(v) for v in origin.get("rpy").split()]
    out = np.eye(4)
    out[:3, :3] = rpy_matrix(*rpy)
    out[:3, 3] = xyz
    return out


def load_urdf_edges(path: Path) -> dict[str, list[tuple[str, str, np.ndarray]]]:
    root = ET.parse(path).getroot()
    edges: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.get("link")
        child_link = child.get("link")
        if not parent_link or not child_link:
            continue
        T = origin_transform(joint.find("origin"))
        edges.setdefault(parent_link, []).append((child_link, joint.get("name", ""), T))
    return edges


def urdf_transform(path: Path, source_link: str, target_link: str) -> tuple[np.ndarray, list[str]]:
    edges = load_urdf_edges(path)
    q = deque([(source_link, np.eye(4), [])])
    seen = {source_link}
    while q:
        link, T, names = q.popleft()
        if link == target_link:
            return T, names
        for child, joint_name, T_child in edges.get(link, []):
            if child in seen:
                continue
            seen.add(child)
            q.append((child, T @ T_child, names + [joint_name]))
    raise RuntimeError(f"No URDF chain from {source_link!r} to {target_link!r}")


def gf_to_np(mat) -> np.ndarray:
    arr = np.array([[mat[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
    return arr.T


def usd_relative_transform(path: Path, parent_prim: str, child_prim: str) -> np.ndarray:
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Cannot open USD: {path}")
    parent = stage.GetPrimAtPath(parent_prim)
    child = stage.GetPrimAtPath(child_prim)
    if not parent or not parent.IsValid():
        raise RuntimeError(f"USD parent prim not found: {parent_prim}")
    if not child or not child.IsValid():
        raise RuntimeError(f"USD camera prim not found: {child_prim}")
    cache = UsdGeom.XformCache()
    parent_world = gf_to_np(cache.GetLocalToWorldTransform(parent))
    child_world = gf_to_np(cache.GetLocalToWorldTransform(child))
    return np.linalg.inv(parent_world) @ child_world


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = a[:3, :3].T @ b[:3, :3]
    cos_theta = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def fmt_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x): .6f}" for x in v) + "]"


def print_transform(label: str, T: np.ndarray) -> None:
    print(label)
    print(f"  translation: {fmt_vec(T[:3, 3])} m")
    print(f"  distance:    {np.linalg.norm(T[:3, 3]):.6f} m")
    print("  rotation:")
    for row in T[:3, :3]:
        print(f"    {fmt_vec(row)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usd", type=Path, default=DEFAULT_USD)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--urdf-parent-link", default="flange")
    p.add_argument("--urdf-camera-link", default="camera_optical_frame")
    p.add_argument("--usd-parent-prim", default=DEFAULT_USD_PARENT)
    p.add_argument("--usd-camera-prim", default=DEFAULT_USD_CAMERA)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    T_urdf, chain = urdf_transform(
        args.urdf,
        args.urdf_parent_link,
        args.urdf_camera_link,
    )
    T_usd = usd_relative_transform(args.usd, args.usd_parent_prim, args.usd_camera_prim)

    print(f"URDF chain: {' -> '.join(chain)}")
    print_transform(
        f"URDF {args.urdf_parent_link} -> {args.urdf_camera_link}",
        T_urdf,
    )
    print_transform(
        f"USD  {args.usd_parent_prim} -> {args.usd_camera_prim}",
        T_usd,
    )

    dt = T_usd[:3, 3] - T_urdf[:3, 3]
    print("Difference, USD - URDF")
    print(f"  translation: {fmt_vec(dt)} m")
    print(f"  norm:        {np.linalg.norm(dt):.6f} m")
    print(f"  rotation:    {rotation_error_deg(T_urdf, T_usd):.3f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())

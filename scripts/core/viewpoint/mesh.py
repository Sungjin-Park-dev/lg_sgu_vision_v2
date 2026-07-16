"""Mesh and material loading for viewpoint generation."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from common import config

def parse_mtl_file(mtl_path: str) -> Dict[str, Dict]:
    """Parse MTL file to extract material properties"""
    materials = {}
    current_material = None

    if not os.path.exists(mtl_path):
        print(f"Warning: MTL file not found: {mtl_path}")
        return materials

    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 0:
                continue

            command = parts[0]

            if command == 'newmtl':
                if len(parts) >= 2:
                    current_material = ' '.join(parts[1:])
                    materials[current_material] = {}
            elif command == 'Kd' and current_material:
                if len(parts) >= 4:
                    try:
                        r = float(parts[1])
                        g = float(parts[2])
                        b = float(parts[3])
                        materials[current_material]['Kd'] = np.array([r, g, b], dtype=np.float64)
                    except ValueError:
                        pass

    return materials


def parse_obj_material_usage(obj_path: str) -> Tuple[Dict[int, str], str]:
    """Parse OBJ file to determine which material is used for each triangle"""
    triangle_materials = {}
    current_material = None
    face_index = 0
    mtl_file = None

    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 0:
                continue

            command = parts[0]

            if command == 'mtllib':
                if len(parts) >= 2:
                    mtl_filename = ' '.join(parts[1:])
                    obj_dir = os.path.dirname(obj_path)
                    mtl_file = os.path.join(obj_dir, mtl_filename)
            elif command == 'usemtl':
                if len(parts) >= 2:
                    current_material = ' '.join(parts[1:])
            elif command == 'f':
                if current_material:
                    triangle_materials[face_index] = current_material
                face_index += 1

    return triangle_materials, mtl_file


def rgb_to_kd(r: int, g: int, b: int) -> np.ndarray:
    """Convert RGB (0-255) to Kd (0.0-1.0)"""
    return np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float64)


def kd_to_rgb(kd: np.ndarray) -> Tuple[int, int, int]:
    """Convert Kd (0.0-1.0) to RGB (0-255)"""
    r = int(round(kd[0] * 255))
    g = int(round(kd[1] * 255))
    b = int(round(kd[2] * 255))
    return (r, g, b)


def match_material_by_color(
    materials: Dict[str, Dict],
    target_rgb: Tuple[int, int, int],
    tolerance: float = 5.0
) -> List[str]:
    """Find materials matching target RGB color within tolerance"""
    target_kd = rgb_to_kd(*target_rgb)
    matched_materials = []

    for mat_name, mat_props in materials.items():
        if 'Kd' not in mat_props:
            continue

        mat_kd = mat_props['Kd']
        mat_rgb = kd_to_rgb(mat_kd)
        distance = np.sqrt(
            (mat_rgb[0] - target_rgb[0])**2 +
            (mat_rgb[1] - target_rgb[1])**2 +
            (mat_rgb[2] - target_rgb[2])**2
        )

        if distance <= tolerance:
            matched_materials.append(mat_name)

    return matched_materials


def extract_target_mesh(
    mesh: trimesh.Trimesh,
    triangle_materials: Dict[int, str],
    target_materials: List[str]
) -> trimesh.Trimesh:
    """Extract submesh containing only triangles with target materials"""
    target_face_indices = []

    for face_idx in range(len(mesh.faces)):
        if face_idx in triangle_materials:
            mat_name = triangle_materials[face_idx]
            if mat_name in target_materials:
                target_face_indices.append(face_idx)

    if len(target_face_indices) == 0:
        raise ValueError(
            f"No triangles found with target materials: {target_materials}\n"
            f"Available materials: {set(triangle_materials.values())}"
        )

    target_mesh = mesh.submesh([target_face_indices], append=True)
    return target_mesh


# ============================================================================
# Path & PCA Utilities
# ============================================================================

def load_meshes(object_name, material_rgb=None, color_tolerance=5.0):
    """소스 메시 로드 + (선택) 재질 RGB 필터로 타깃 메시 추출.

    Returns: (full_mesh, target_mesh, input_path)
    """
    input_path = str(config.get_mesh_path(object_name, mesh_type="source"))
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input mesh not found: {input_path}")

    print("Loading mesh...")
    loaded = trimesh.load(input_path)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded
    print(f"  Loaded: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} triangles")
    print()

    if material_rgb:
        print("Parsing materials...")
        triangle_materials, mtl_file = parse_obj_material_usage(input_path)
        if mtl_file is None or not os.path.exists(mtl_file):
            raise FileNotFoundError("MTL file not found")

        materials = parse_mtl_file(mtl_file)
        print(f"  Found {len(materials)} materials:")
        for mat_name, mat_props in materials.items():
            if 'Kd' in mat_props:
                rgb = kd_to_rgb(mat_props['Kd'])
                print(f"    - {mat_name}: RGB{rgb}")
        print()

        print("Matching material...")
        target_rgb = tuple(map(int, material_rgb.split(',')))
        matched_materials = match_material_by_color(materials, target_rgb, color_tolerance)
        if len(matched_materials) == 0:
            raise ValueError(
                f"No materials matched RGB{target_rgb} within tolerance {color_tolerance}")
        print(f"  Matched: {matched_materials}")
        print()

        print("Extracting target mesh...")
        target_mesh = extract_target_mesh(mesh, triangle_materials, matched_materials)
        target_percentage = (len(target_mesh.faces) / len(mesh.faces)) * 100
        print(f"  Target: {len(target_mesh.faces):,} / {len(mesh.faces):,} triangles ({target_percentage:.1f}%)")
    else:
        print("Using entire mesh (no material filter)...")
        target_mesh = mesh
        print(f"  Triangles: {len(target_mesh.faces):,}")

    print(f"  Surface area: {target_mesh.area:.6f} m2")
    print()
    return mesh, target_mesh, input_path

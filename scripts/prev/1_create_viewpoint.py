#!/usr/bin/env python3
"""
메시에서 뷰포인트 생성

통합 스크립트 기능:
- 다중 재질 OBJ 전처리 (선택 사항)
- 포아송 디스크 샘플링을 통한 균일한 뷰포인트 생성

입력: OBJ 파일 (선택적으로 재질 RGB 색상 필터 적용 가능)
출력: 표면 위치와 법선 벡터가 포함된 HDF5 파일

사용법:
    # 특정 재질만 선택하여 뷰포인트 생성
    omni_python scripts/1_create_viewpoint.py \
        --object sample \
        --material-rgb "170,163,158" \
        --visualize

    # 전체 메시에서 뷰포인트 생성
    omni_python scripts/1_create_viewpoint.py \
        --object sample
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import trimesh
import open3d as o3d
import h5py

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from common import config


# ============================================================================
# MTL/OBJ Parsing Functions (from scripts/preprocess_mesh.py)
# ============================================================================

def parse_mtl_file(mtl_path: str) -> Dict[str, Dict]:
    """
    Parse MTL file to extract material properties

    Args:
        mtl_path: Path to MTL file

    Returns:
        Dictionary mapping material names to properties:
        {
            "material_name": {
                "Kd": np.array([r, g, b])  # Diffuse color in [0,1]
            }
        }
    """
    materials = {}
    current_material = None

    if not os.path.exists(mtl_path):
        print(f"Warning: MTL file not found: {mtl_path}")
        return materials

    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 0:
                continue

            command = parts[0]

            # New material definition
            if command == 'newmtl':
                if len(parts) >= 2:
                    current_material = ' '.join(parts[1:])
                    materials[current_material] = {}

            # Diffuse color
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
    """
    Parse OBJ file to determine which material is used for each triangle

    Args:
        obj_path: Path to OBJ file

    Returns:
        triangle_materials: Dict mapping triangle index to material name
        mtl_file: Path to MTL file
    """
    triangle_materials = {}
    current_material = None
    face_index = 0
    mtl_file = None

    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 0:
                continue

            command = parts[0]

            # MTL library reference
            if command == 'mtllib':
                if len(parts) >= 2:
                    mtl_filename = ' '.join(parts[1:])
                    obj_dir = os.path.dirname(obj_path)
                    mtl_file = os.path.join(obj_dir, mtl_filename)

            # Material usage
            elif command == 'usemtl':
                if len(parts) >= 2:
                    current_material = ' '.join(parts[1:])

            # Face definition
            elif command == 'f':
                if current_material:
                    triangle_materials[face_index] = current_material
                face_index += 1

    return triangle_materials, mtl_file


def rgb_to_kd(r: int, g: int, b: int) -> np.ndarray:
    """
    Convert RGB (0-255) to Kd (0.0-1.0)

    Args:
        r, g, b: RGB values in range [0, 255]

    Returns:
        Kd values as numpy array [0.0, 1.0]
    """
    return np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float64)


def kd_to_rgb(kd: np.ndarray) -> Tuple[int, int, int]:
    """
    Convert Kd (0.0-1.0) to RGB (0-255)

    Args:
        kd: Kd values as numpy array [0.0, 1.0]

    Returns:
        Tuple of (r, g, b) in range [0, 255]
    """
    r = int(round(kd[0] * 255))
    g = int(round(kd[1] * 255))
    b = int(round(kd[2] * 255))
    return (r, g, b)


def match_material_by_color(
    materials: Dict[str, Dict],
    target_rgb: Tuple[int, int, int],
    tolerance: float = 5.0
) -> List[str]:
    """
    Find materials matching target RGB color within tolerance

    Args:
        materials: Material dictionary from parse_mtl_file()
        target_rgb: Target RGB color tuple (0-255)
        tolerance: Maximum color distance for matching

    Returns:
        List of matching material names
    """
    target_kd = rgb_to_kd(*target_rgb)
    matched_materials = []

    for mat_name, mat_props in materials.items():
        if 'Kd' not in mat_props:
            continue

        mat_kd = mat_props['Kd']

        # Calculate Euclidean distance in RGB space (0-255)
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
    """
    Extract submesh containing only triangles with target materials

    Args:
        mesh: Input Trimesh object
        triangle_materials: Dict mapping triangle index to material name
        target_materials: List of material names to extract

    Returns:
        New Trimesh containing only target triangles
    """
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

    # Extract submesh
    target_mesh = mesh.submesh([target_face_indices], append=True)

    return target_mesh


# ============================================================================
# Viewpoint Generation Functions
# ============================================================================

def calculate_required_viewpoints(surface_area_m2: float) -> int:
    """
    Calculate required number of viewpoints based on surface area and camera FOV

    Formula: N = ceil(area / (fov_w * fov_h * (1 - overlap)^2))

    Args:
        surface_area_m2: Surface area in square meters

    Returns:
        Required number of viewpoints
    """
    # Convert FOV from mm to m
    fov_w_m = config.CAMERA_FOV_WIDTH_MM / 1000.0
    fov_h_m = config.CAMERA_FOV_HEIGHT_MM / 1000.0
    overlap = config.CAMERA_OVERLAP_RATIO

    # Calculate effective area per viewpoint (accounting for overlap)
    effective_w = fov_w_m * (1.0 - overlap)
    effective_h = fov_h_m * (1.0 - overlap)
    effective_area_per_viewpoint = effective_w * effective_h

    # Calculate required viewpoints
    num_viewpoints = int(np.ceil(surface_area_m2 / effective_area_per_viewpoint))

    # Ensure at least 1 viewpoint
    return max(1, num_viewpoints)


def sample_viewpoints_poisson(
    mesh: trimesh.Trimesh,
    num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample viewpoints uniformly using Poisson disk sampling

    Args:
        mesh: Input Trimesh object
        num_points: Number of points to sample

    Returns:
        positions: (N, 3) array of surface positions
        normals: (N, 3) array of surface normals (unit vectors)
    """
    # Convert Trimesh to Open3D
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()

    try:
        # Try Poisson disk sampling
        pcd = o3d_mesh.sample_points_poisson_disk(
            number_of_points=num_points,
            init_factor=5
        )

        # Extract positions
        positions = np.asarray(pcd.points, dtype=np.float32)

        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.01, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=10)

        normals = np.asarray(pcd.normals, dtype=np.float32)

    except Exception as e:
        print(f"Warning: Poisson disk sampling failed ({e}), falling back to uniform sampling")

        # Fallback to uniform sampling
        pcd = o3d_mesh.sample_points_uniformly(number_of_points=num_points)

        positions = np.asarray(pcd.points, dtype=np.float32)

        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.01, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=10)

        normals = np.asarray(pcd.normals, dtype=np.float32)

    return positions, normals


# ============================================================================
# HDF5 I/O Functions
# ============================================================================

def save_viewpoints_hdf5(
    positions: np.ndarray,
    normals: np.ndarray,
    output_path: str,
    metadata: Optional[dict] = None,
    camera_spec: Optional[dict] = None
) -> Path:
    """
    Save viewpoints to HDF5 file

    Args:
        positions: (N, 3) array of surface positions in meters
        normals: (N, 3) array of surface normals (unit vectors)
        output_path: Path to output HDF5 file
        metadata: Optional metadata dictionary
        camera_spec: Optional camera specification dictionary

    Returns:
        Path to saved file
    """
    if positions.shape != normals.shape:
        raise ValueError(
            f"Positions and normals must have same shape, "
            f"got {positions.shape} and {normals.shape}"
        )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Positions must be (N, 3) array, got shape {positions.shape}"
        )

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Create viewpoints group
        viewpoints_grp = f.create_group('viewpoints')
        viewpoints_grp.create_dataset('positions', data=positions.astype(np.float32))
        viewpoints_grp.create_dataset('normals', data=normals.astype(np.float32))

        # Create metadata group
        metadata_grp = f.create_group('metadata')
        metadata_grp.attrs['num_viewpoints'] = len(positions)

        # Add additional metadata
        if metadata:
            for key, value in metadata.items():
                if key != 'camera_spec':  # Handle separately
                    metadata_grp.attrs[key] = value

        # Add camera spec
        if camera_spec:
            camera_spec_grp = metadata_grp.create_group('camera_spec')
            for key, value in camera_spec.items():
                camera_spec_grp.attrs[key] = value

    print(f"  ✓ Saved {len(positions)} viewpoints to {output_path}")

    return output_path


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_viewpoints(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    normals: np.ndarray
):
    """
    Visualize mesh and viewpoints using Open3D

    Args:
        mesh: Target mesh
        positions: Surface positions
        normals: Surface normals
    """
    print("\nOpening visualization...")
    print("  Gray mesh: Target surface")
    print("  Red points: Surface sample positions")
    print("  Green points: Camera positions (offset by working distance)")

    # Convert Trimesh to Open3D
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    o3d_mesh.compute_vertex_normals()

    # Create surface viewpoint point cloud (red)
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(positions)
    surface_pcd.paint_uniform_color([1, 0, 0])

    # Create camera positions (offset by working distance) (green)
    working_distance_m = config.CAMERA_WORKING_DISTANCE_MM / 1000.0
    camera_positions = positions + normals * working_distance_m
    camera_pcd = o3d.geometry.PointCloud()
    camera_pcd.points = o3d.utility.Vector3dVector(camera_positions)
    camera_pcd.paint_uniform_color([0, 1, 0])

    # Visualize
    o3d.visualization.draw_geometries(
        [o3d_mesh, surface_pcd, camera_pcd],
        window_name="Viewpoint Visualization",
        width=1280,
        height=720
    )


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create viewpoints from OBJ mesh',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 전체 메시에서 뷰포인트 생성
  python %(prog)s --object sample

  # 특정 material만 선택하여 뷰포인트 생성
  python %(prog)s --object sample --material-rgb "0,255,0"

  # 시각화 포함
  python %(prog)s --object sample --material-rgb "0,255,0" --visualize

Note:
  - Number of viewpoints is calculated automatically based on surface area and camera FOV
  - Input mesh:  data/{object}/mesh/source.obj
  - Output file: data/{object}/viewpoint/{calculated_num}/viewpoints.h5
        """
    )

    # Required arguments
    parser.add_argument(
        '--object',
        type=str,
        required=True,
        help='Object name for auto-path generation (e.g., "sample", "glass")'
    )

    # Optional arguments
    parser.add_argument(
        '--material-rgb',
        type=str,
        default=None,
        help='Target material RGB color as "R,G,B" (e.g., "0,255,0"). If not specified, use entire mesh.'
    )
    parser.add_argument(
        '--color-tolerance',
        type=float,
        default=5.0,
        help='RGB color matching tolerance (default: 5.0)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show Open3D visualization after generation'
    )

    args = parser.parse_args()

    # Validate RGB format if provided
    if args.material_rgb is not None:
        try:
            rgb_parts = args.material_rgb.split(',')
            if len(rgb_parts) != 3:
                raise ValueError("RGB must have 3 components")
            r, g, b = map(int, rgb_parts)
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                raise ValueError("RGB values must be in range [0, 255]")
        except ValueError as e:
            parser.error(f"Invalid RGB format: {e}")

    return args


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main execution function"""
    args = parse_arguments()

    # Resolve input path (output path will be determined after calculating num_viewpoints)
    # Use "source" mesh which contains multi-material information
    input_path = str(config.get_mesh_path(args.object, mesh_type="source"))

    print("=" * 60)
    print("CREATE VIEWPOINTS")
    print("=" * 60)
    print(f"Object: {args.object}")
    print(f"Input:  {input_path}")
    if args.material_rgb:
        print(f"Target RGB: {args.material_rgb}")
    else:
        print(f"Target: 전체 메시 (material 필터 없음)")
    print()

    # Validate input exists
    if not os.path.exists(input_path):
        print(f"Error: Input mesh not found: {input_path}")
        return 1

    # 1. Load OBJ
    print("Loading mesh...")
    loaded = trimesh.load(input_path)

    # Handle Scene (multi-material) or single Trimesh
    if isinstance(loaded, trimesh.Scene):
        # Concatenate all geometries in the scene into a single mesh
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded

    print(f"  Loaded: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} triangles")
    print()

    # 2. Determine target mesh (filter by material or use entire mesh)
    if args.material_rgb:
        # Filter by material RGB
        print("Parsing materials...")
        triangle_materials, mtl_file = parse_obj_material_usage(input_path)

        if mtl_file is None or not os.path.exists(mtl_file):
            print(f"Error: MTL file not found")
            return 1

        materials = parse_mtl_file(mtl_file)
        print(f"  Found {len(materials)} materials:")
        for mat_name, mat_props in materials.items():
            if 'Kd' in mat_props:
                rgb = kd_to_rgb(mat_props['Kd'])
                print(f"    - {mat_name}: RGB{rgb}")
        print()

        # Match material by RGB
        print("Matching material...")
        target_rgb = tuple(map(int, args.material_rgb.split(',')))
        matched_materials = match_material_by_color(materials, target_rgb, args.color_tolerance)

        if len(matched_materials) == 0:
            print(f"  Error: No materials matched RGB{target_rgb} within tolerance {args.color_tolerance}")
            print(f"  Available materials listed above")
            return 1

        print(f"  Matched: {matched_materials}")
        print()

        # Extract target mesh
        print("Extracting target mesh...")
        target_mesh = extract_target_mesh(mesh, triangle_materials, matched_materials)
        target_percentage = (len(target_mesh.faces) / len(mesh.faces)) * 100
        print(f"  Target: {len(target_mesh.faces):,} / {len(mesh.faces):,} triangles ({target_percentage:.1f}%)")
    else:
        # Use entire mesh
        print("Using entire mesh (no material filter)...")
        target_mesh = mesh
        print(f"  Triangles: {len(target_mesh.faces):,}")

    print(f"  Surface area: {target_mesh.area:.6f} m²")
    print()

    # 5. Calculate required viewpoints
    print("Calculating viewpoints...")
    num_viewpoints = calculate_required_viewpoints(target_mesh.area)
    print(f"  FOV: {config.CAMERA_FOV_WIDTH_MM} × {config.CAMERA_FOV_HEIGHT_MM} mm")
    print(f"  Overlap: {config.CAMERA_OVERLAP_RATIO * 100:.0f}%")
    print(f"  Required: {num_viewpoints} viewpoints")
    print()

    # 6. Sample viewpoints
    print("Sampling (Poisson disk)...")
    positions, normals = sample_viewpoints_poisson(target_mesh, num_viewpoints)
    print(f"  Sampled: {len(positions)} viewpoints")
    print()

    # 7. Resolve output path using calculated num_viewpoints
    output_path = str(config.get_viewpoint_path(args.object, len(positions)))
    print(f"Output: {output_path}")
    print()

    # 8. Save to HDF5
    print("Saving to HDF5...")
    camera_spec = {
        'fov_width_mm': config.CAMERA_FOV_WIDTH_MM,
        'fov_height_mm': config.CAMERA_FOV_HEIGHT_MM,
        'working_distance_mm': config.CAMERA_WORKING_DISTANCE_MM,
        'overlap_ratio': config.CAMERA_OVERLAP_RATIO,
    }
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_mesh': str(input_path),
    }
    save_viewpoints_hdf5(positions, normals, output_path, metadata, camera_spec)
    print()

    print("Complete!")
    print("=" * 60)
    print()

    # 9. Optional visualization
    if args.visualize:
        visualize_viewpoints(target_mesh, positions, normals)

    return 0


if __name__ == '__main__':
    sys.exit(main())

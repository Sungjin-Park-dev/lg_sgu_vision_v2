#!/usr/bin/env python3
"""
메시 표면 법선 벡터 생성 + 시각화 (+ 선택적 CoACD 분할)

물체 파일(.stl/.obj/.ply 등)을 읽어 표면을 균일하게 샘플링하고, 각 샘플 점이
속한 면의 법선을 구한다. 결과는 인터랙티브 HTML로 시각화하고 HDF5로 저장한다.
--coacd 옵션을 주면 메시를 convex 파트로 분할하여 함께 시각화/저장한다.

generate_viewpoints.py에서 법선 생성/시각화/CoACD 분할 기능만 추려낸 독립 실행 버전.
(PCA 그리드, 지그재그 경로, DBSCAN/GTSP 클러스터링, 재질 필터 등은 모두 제외)

사용법:
    # 기본: 표면 샘플링 법선 생성 + HTML 시각화 + HDF5 저장
    uv run scripts/prep/generate_normals.py --input data/sample/mesh/source.obj

    # 출력 경로 지정 + 샘플 수 조절
    uv run scripts/prep/generate_normals.py --input model.stl --output out/model --num-samples 20000

    # CoACD convex decomposition 으로 물체 분할까지
    uv run scripts/prep/generate_normals.py --input model.obj --coacd --coacd-threshold 0.2
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import trimesh
import h5py


# ============================================================================
# 색상 팔레트 (CoACD 파트 시각화용)
# ============================================================================

_PART_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#469990', '#9A6324',
    '#800000', '#aaffc3', '#808000', '#000075', '#dcbeff',
    '#ffe119', '#ff6347', '#00ced1', '#ff1493', '#7fff00',
]


# ============================================================================
# 메시 로드
# ============================================================================

def load_mesh(input_path: str) -> trimesh.Trimesh:
    """메시 파일을 로드한다. Scene이면 단일 메시로 합친다."""
    loaded = trimesh.load(input_path)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded
    return mesh


# ============================================================================
# 법선 생성 (표면 균일 샘플링)
# ============================================================================

def generate_surface_normals(
    mesh: trimesh.Trimesh,
    num_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """메시 표면을 균일 샘플링하고, 각 점이 속한 면의 법선을 반환한다.

    Args:
        mesh: 대상 메시
        num_samples: 표면 샘플 점 개수
        seed: 재현성을 위한 난수 시드

    Returns:
        positions: (N, 3) float32 — 표면 위 샘플 점
        normals:   (N, 3) float32 — 각 점의 단위 법선 벡터
    """
    np.random.seed(seed)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[face_indices]

    # 단위 벡터 정규화 (0-길이 법선 방어)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normals = normals / norms

    return points.astype(np.float32), normals.astype(np.float32)


# ============================================================================
# CoACD 분할
# ============================================================================

def decompose_coacd(
    mesh: trimesh.Trimesh,
    threshold: float = 0.2,
) -> List[trimesh.Trimesh]:
    """CoACD convex decomposition 으로 메시를 convex 파트로 분할한다.

    Args:
        mesh: 대상 메시
        threshold: concavity threshold (낮을수록 더 많은 파트)

    Returns:
        part_meshes: convex 파트 메시 목록
    """
    import coacd

    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh, threshold=threshold)
    print(f"  CoACD: {len(parts)} convex parts")

    part_meshes = [
        trimesh.Trimesh(vertices=verts, faces=faces) for verts, faces in parts
    ]
    return part_meshes


def assign_points_to_parts(
    part_meshes: List[trimesh.Trimesh],
    positions: np.ndarray,
) -> np.ndarray:
    """각 샘플 점을 가장 가까운 CoACD 파트에 할당한다.

    Args:
        part_meshes: convex 파트 메시 목록
        positions: (N, 3) 표면 샘플 점

    Returns:
        point_part_ids: (N,) 각 점이 속한 파트 인덱스
    """
    distances = np.full((len(positions), len(part_meshes)), np.inf)
    for j, part in enumerate(part_meshes):
        _, dists, _ = trimesh.proximity.closest_point(part, positions)
        distances[:, j] = dists
    return np.argmin(distances, axis=1).astype(np.int32)


# ============================================================================
# 시각화 (Plotly HTML)
# ============================================================================

def visualize_normals_html(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    output_html: str,
    max_display: int = 5000,
    part_meshes: Optional[List[trimesh.Trimesh]] = None,
    point_part_ids: Optional[np.ndarray] = None,
):
    """법선 샘플 점을 인터랙티브 3D HTML로 저장한다 (generate_viewpoints 스타일).

    법선은 화살표 대신 큰 점 마커(size=5, 검은 외곽선)로 표현한다.
    CoACD 분할 시(part_meshes/point_part_ids 제공):
      - convex 파트 메시를 _PART_COLORS 색으로 반투명(opacity 0.3) 표시
      - 각 점을 소속 파트 색으로 칠하고, 파트별로 범례 토글 가능
    """
    import plotly.graph_objects as go

    # 표시용 서브샘플링 (저장은 전체, 화면은 일부만)
    n = len(positions)
    if n > max_display:
        sel = np.random.default_rng(0).choice(n, size=max_display, replace=False)
        disp_pos = positions[sel]
        disp_pids = point_part_ids[sel] if point_part_ids is not None else None
        print(f"  HTML: {max_display}/{n} 점만 표시 (저장은 전체)")
    else:
        disp_pos = positions
        disp_pids = point_part_ids

    verts, faces = mesh.vertices, mesh.faces

    # 베이스 메시 (연회색, 반투명)
    traces = [go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='lightgray', opacity=0.25,
        name='Mesh', hoverinfo='skip',
    )]

    # CoACD convex 파트 메시 (파트별 색, 반투명)
    if part_meshes is not None:
        for j, part in enumerate(part_meshes):
            color = _PART_COLORS[j % len(_PART_COLORS)]
            v, f = part.vertices, part.faces
            traces.append(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=color, opacity=0.30,
                name=f'CoACD Part {j}', hoverinfo='name',
                legendgroup=f'part{j}',
            ))

    # 법선 샘플 점 — 큰 마커 (검은 외곽선)
    if disp_pids is None:
        traces.append(go.Scatter3d(
            x=disp_pos[:, 0], y=disp_pos[:, 1], z=disp_pos[:, 2],
            mode='markers',
            marker=dict(size=5, color='#4363d8', line=dict(width=1, color='black')),
            name=f'Points ({n})', hoverinfo='skip',
        ))
    else:
        for pid in np.unique(disp_pids):
            color = _PART_COLORS[pid % len(_PART_COLORS)]
            mask = disp_pids == pid
            p = disp_pos[mask]
            traces.append(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2],
                mode='markers',
                marker=dict(size=5, color=color, line=dict(width=1, color='black')),
                name=f'Part {pid} ({int(mask.sum())} pts)',
                legendgroup=f'part{pid}', hoverinfo='skip',
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Surface Normals ({n} points)',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
        ),
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig.write_html(output_html)
    print(f"  법선 시각화 저장: {output_html}")


# ============================================================================
# HDF5 저장
# ============================================================================

def save_normals_hdf5(
    positions: np.ndarray,
    normals: np.ndarray,
    output_path: str,
    metadata: Optional[dict] = None,
) -> Path:
    """법선(위치+벡터)을 HDF5로 저장한다.

    구조는 기존 viewpoints.h5 와 호환 (viewpoints/positions, viewpoints/normals).
    """
    if positions.shape != normals.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions/normals 는 동일한 (N, 3) 형태여야 합니다. "
            f"got {positions.shape}, {normals.shape}"
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, 'w') as f:
        grp = f.create_group('viewpoints')
        grp.create_dataset('positions', data=positions.astype(np.float32))
        grp.create_dataset('normals', data=normals.astype(np.float32))

        meta = f.create_group('metadata')
        meta.attrs['num_points'] = len(positions)
        if metadata:
            for k, v in metadata.items():
                meta.attrs[k] = v

    print(f"  법선 {len(positions)}개 저장: {out}")
    return out


# ============================================================================
# CLI
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='메시 표면 법선 생성 + 시각화 (+ 선택적 CoACD 분할)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input', type=str, required=True,
                        help='입력 메시 파일 (.stl/.obj/.ply 등)')
    parser.add_argument('--output', type=str, default=None,
                        help='출력 기본 경로 (확장자 제외). 기본: 입력 파일과 동일 위치/이름')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='표면 샘플 점 개수 (기본: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='샘플링 난수 시드 (기본: 42)')
    parser.add_argument('--max-display', type=int, default=5000,
                        help='HTML에 표시할 최대 점 개수 (기본: 5000, 저장은 전체)')
    parser.add_argument('--no-html', action='store_true',
                        help='HTML 시각화 생성 건너뛰기')

    parser.add_argument('--coacd', action='store_true',
                        help='CoACD convex decomposition 으로 물체 분할 수행')
    parser.add_argument('--coacd-threshold', type=float, default=0.2,
                        help='[coacd] concavity threshold (낮을수록 더 많은 파트, 기본: 0.2)')

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: 입력 메시를 찾을 수 없습니다: {input_path}")
        return 1

    # 출력 기본 경로 결정 (확장자 제외 stem)
    if args.output:
        out_base = Path(args.output)
        if out_base.suffix:  # 확장자가 붙어 있으면 제거
            out_base = out_base.with_suffix('')
    else:
        out_base = Path(input_path).with_suffix('')
    out_base.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATE NORMALS")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {out_base}.*")
    print()

    # 1. 메시 로드
    print("Loading mesh...")
    mesh = load_mesh(input_path)
    print(f"  Loaded: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} triangles")
    print(f"  Surface area: {mesh.area:.6f}")
    print()

    # 2. 표면 법선 생성
    print(f"Sampling surface normals ({args.num_samples} points)...")
    positions, normals = generate_surface_normals(mesh, args.num_samples, args.seed)
    print(f"  Generated: {len(positions)} points + normals")
    print()

    # 3. HDF5 저장
    print("Saving normals to HDF5...")
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_mesh': str(input_path),
        'num_samples': args.num_samples,
        'seed': args.seed,
        'method': 'surface_sampling',
    }
    save_normals_hdf5(positions, normals, f"{out_base}.h5", metadata)
    print()

    # 4. CoACD 분할 (옵션) — 법선 시각화보다 먼저 실행하여 점에 파트 색을 입힌다
    part_meshes = None
    point_part_ids = None
    if args.coacd:
        print(f"Decomposing with CoACD (threshold={args.coacd_threshold})...")
        part_meshes = decompose_coacd(mesh, args.coacd_threshold)

        # 각 샘플 점을 가장 가까운 파트에 할당 (HTML 점/파트 색 입힘에만 사용)
        point_part_ids = assign_points_to_parts(part_meshes, positions)
        print()

    # 5. 법선 시각화 — HTML은 항상 하나. CoACD 적용 시 점/파트가 파트 색으로 표시된다.
    if not args.no_html:
        print("Visualizing normals...")
        visualize_normals_html(
            mesh, positions, f"{out_base}.html",
            args.max_display,
            part_meshes=part_meshes,
            point_part_ids=point_part_ids,
        )
        print()

    print("Complete!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())

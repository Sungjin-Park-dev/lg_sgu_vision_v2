#!/usr/bin/env python3
"""
CoACD Convex Decomposition 시각화

메시를 CoACD로 convex 파트로 분해하고, 각 파트를 색상별로 HTML로 시각화한다.

사용법:
    # 기본 (source mesh)
    uv run scripts/visualize_coacd.py --object sample

    # threshold 조절
    uv run scripts/visualize_coacd.py --object sample --threshold 0.05

    # 재질 필터 (target mesh)
    uv run scripts/visualize_coacd.py --object sample --material-rgb "170,163,158"

    # threshold 비교
    uv run scripts/visualize_coacd.py --object sample --material-rgb "170,163,158" --compare

    uv run scripts/visualize_coacd.py --object glass --compare

"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import trimesh
import coacd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import config


_BOLD_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
    '#000075', '#a9a9a9', '#e6beff', '#ffe119', '#ffd8b1',
    '#fffac8', '#ff6347', '#00ced1', '#ff1493', '#7fff00',
]


def run_coacd(mesh: trimesh.Trimesh, threshold: float) -> list:
    """CoACD 실행 후 trimesh 리스트 반환."""
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh, threshold=threshold)
    part_meshes = []
    for verts, faces in parts:
        part_meshes.append(trimesh.Trimesh(vertices=verts, faces=faces))
    return part_meshes


def build_traces(part_meshes: list, label: str):
    """파트 메시 리스트에서 plotly traces 생성."""
    traces = []
    for j, pm in enumerate(part_meshes):
        color = _BOLD_COLORS[j % len(_BOLD_COLORS)]
        v, f = pm.vertices, pm.faces
        traces.append(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=color, opacity=0.6,
            name=f'Part {j} ({len(f)} faces)',
            hoverinfo='name',
            legendgroup=f'{label}_p{j}',
        ))
    return traces


def build_html(mesh, all_results: dict, output_html: str):
    """결과를 HTML로 저장. all_results: {label: part_meshes}"""
    # 원본 메시 (반투명 회색)
    v, f = mesh.vertices, mesh.faces
    base_trace = go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color='lightgray', opacity=0.15,
        name='Original', hoverinfo='skip',
    )

    labels = list(all_results.keys())

    # 모든 모드의 traces 수집
    fig_traces = [base_trace]
    mode_trace_ranges = []
    for label in labels:
        start = len(fig_traces)
        fig_traces.extend(build_traces(all_results[label], label))
        mode_trace_ranges.append((start, len(fig_traces)))

    fig = go.Figure(data=fig_traces)

    # visibility 초기화: base + 첫 번째 모드만 표시
    for i, trace in enumerate(fig.data):
        if i == 0:
            trace.visible = True
        elif mode_trace_ranges[0][0] <= i < mode_trace_ranges[0][1]:
            trace.visible = True
        else:
            trace.visible = False

    # 드롭다운
    if len(labels) > 1:
        buttons = []
        for mode_idx, label in enumerate(labels):
            n_parts = len(all_results[label])
            total_faces = sum(len(pm.faces) for pm in all_results[label])
            button_label = f"{label} ({n_parts} parts, {total_faces} faces)"

            visibility = [True] + [False] * (len(fig_traces) - 1)
            s, e = mode_trace_ranges[mode_idx]
            for j in range(s, e):
                visibility[j] = True

            buttons.append(dict(
                label=button_label,
                method='update',
                args=[{'visible': visibility}],
            ))

        fig.update_layout(
            updatemenus=[dict(
                type='dropdown', direction='down',
                x=0.01, xanchor='left', y=1.15, yanchor='top',
                buttons=buttons, font=dict(size=13), bgcolor='white',
            )],
        )

    fig.update_layout(
        title=f'CoACD Decomposition ({len(mesh.faces)} faces)',
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
        ),
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    print(f"  HTML saved to {output_html}")


def load_mesh(args):
    """메시 로드 (재질 필터 지원)."""
    # generate_viewpoints.py의 로직 재사용
    from scripts.generate_viewpoints import (
        parse_mtl_file, parse_obj_material_usage,
        match_material_by_color, extract_target_mesh, kd_to_rgb,
    )

    input_path = str(config.get_mesh_path(args.object, mesh_type="source"))
    print(f"Loading: {input_path}")

    loaded = trimesh.load(input_path)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded
    print(f"  {len(mesh.vertices):,} vertices, {len(mesh.faces):,} triangles")

    if args.material_rgb:
        triangle_materials, mtl_file = parse_obj_material_usage(input_path)
        materials = parse_mtl_file(mtl_file)
        target_rgb = tuple(map(int, args.material_rgb.split(',')))
        matched_materials = match_material_by_color(materials, target_rgb, args.color_tolerance)
        if not matched_materials:
            print(f"  Error: No materials matched RGB{target_rgb}")
            sys.exit(1)
        mesh = extract_target_mesh(mesh, triangle_materials, matched_materials)
        print(f"  Filtered: {len(mesh.faces):,} triangles (material RGB{target_rgb})")

    return mesh, input_path


def main():
    parser = argparse.ArgumentParser(description='CoACD Convex Decomposition 시각화')
    parser.add_argument('--object', type=str, required=True, help='오브젝트 이름')
    parser.add_argument('--material-rgb', type=str, default=None,
                        help='Target material RGB (e.g. "170,163,158")')
    parser.add_argument('--color-tolerance', type=float, default=5.0)
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='CoACD concavity threshold (기본: 0.05)')
    parser.add_argument('--compare', action='store_true',
                        help='여러 threshold 비교')
    args = parser.parse_args()

    mesh, input_path = load_mesh(args)

    if args.compare:
        thresholds = [0.01, 0.03, 0.05, 0.1, 0.3]
        print(f"\nComparing thresholds: {thresholds}")
        all_results = {}
        for t in thresholds:
            print(f"\n  threshold={t}...")
            parts = run_coacd(mesh, t)
            label = f"t={t}"
            print(f"    {len(parts)} parts")
            all_results[label] = parts

        output_html = str(config.get_mesh_path(args.object, mesh_type="source").parent / "coacd_compare.html")
    else:
        print(f"\nCoACD (threshold={args.threshold})...")
        parts = run_coacd(mesh, args.threshold)
        label = f"t={args.threshold}"
        print(f"  {len(parts)} parts")
        all_results = {label: parts}

        output_html = str(config.get_mesh_path(args.object, mesh_type="source").parent / "coacd.html")

    print()
    build_html(mesh, all_results, output_html)


if __name__ == '__main__':
    main()

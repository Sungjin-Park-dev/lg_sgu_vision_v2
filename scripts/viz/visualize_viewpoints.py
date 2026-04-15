"""Plotly HTML 시각화 — 클러스터링된 뷰포인트 경로를 인터랙티브 3D로 출력.

generate_viewpoints.py에서 import하여 사용.
"""

import numpy as np
import plotly.graph_objects as go


# ============================================================================
# Colors
# ============================================================================

_BOLD_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
    '#000075', '#a9a9a9', '#e6beff', '#ffe119', '#ffd8b1',
    '#fffac8', '#ff6347', '#00ced1', '#ff1493', '#7fff00',
]

_PART_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#469990', '#9A6324',
    '#800000', '#aaffc3', '#808000', '#000075', '#dcbeff',
]


# ============================================================================
# Internal helpers
# ============================================================================

def _build_cluster_mode_traces(
    camera_positions, cluster_ids, cluster_order, path_order, mode_label,
    coacd_parts=None, coacd_ids=None,
):
    """한 클러스터링 모드에 대한 plotly traces 생성."""
    K = len(cluster_order)
    colors = [_BOLD_COLORS[i % len(_BOLD_COLORS)] for i in range(K)]

    traces = []

    # CoACD 파트별 색상 메시
    if coacd_parts is not None:
        if coacd_ids is not None:
            # coacd+dbscan: 파트 인덱스 기준으로 독립 색상
            for j, part_mesh in enumerate(coacd_parts):
                color = _PART_COLORS[j % len(_PART_COLORS)]
                v = part_mesh.vertices
                f = part_mesh.faces
                traces.append(go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2],
                    i=f[:, 0], j=f[:, 1], k=f[:, 2],
                    color=color, opacity=0.3,
                    name=f'CoACD Part {j}',
                    hoverinfo='name',
                    legendgroup=f'{mode_label}_part{j}',
                    showlegend=True,
                ))
        else:
            # coacd 단독: 클러스터 방문 순서 색상과 일치
            cid_to_rank = {cid: rank for rank, cid in enumerate(cluster_order)}
            for j, part_mesh in enumerate(coacd_parts):
                rank = cid_to_rank.get(j)
                if rank is None:
                    continue
                color = colors[rank]
                v = part_mesh.vertices
                f = part_mesh.faces
                traces.append(go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2],
                    i=f[:, 0], j=f[:, 1], k=f[:, 2],
                    color=color, opacity=0.3,
                    name=f'Part {j}',
                    hoverinfo='name',
                    legendgroup=f'{mode_label}_c{j}',
                    showlegend=False,
                ))

    for rank, cid in enumerate(cluster_order):
        mask = cluster_ids == cid
        indices = np.where(mask)[0]
        color = colors[rank]
        n_pts = len(indices)

        cam = camera_positions[indices]
        traces.append(go.Scatter3d(
            x=cam[:, 0], y=cam[:, 1], z=cam[:, 2],
            mode='markers',
            marker=dict(size=5, color=color, line=dict(width=1, color='black')),
            name=f'C{cid} ({n_pts} pts)',
            text=[f'vp={idx} cluster={cid}' for idx in indices],
            hoverinfo='text',
            legendgroup=f'{mode_label}_c{cid}',
        ))

        cluster_sorted = sorted(indices, key=lambda i: path_order[i])
        ordered_cam = camera_positions[cluster_sorted]
        if len(ordered_cam) > 1:
            traces.append(go.Scatter3d(
                x=ordered_cam[:, 0], y=ordered_cam[:, 1], z=ordered_cam[:, 2],
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False, hoverinfo='skip',
                legendgroup=f'{mode_label}_c{cid}',
            ))

    # Inter-cluster transitions
    for i in range(len(cluster_order) - 1):
        cid_from, cid_to = cluster_order[i], cluster_order[i + 1]
        from_idx = np.where(cluster_ids == cid_from)[0]
        to_idx = np.where(cluster_ids == cid_to)[0]
        p1 = camera_positions[from_idx[np.argmax(path_order[from_idx])]]
        p2 = camera_positions[to_idx[np.argmin(path_order[to_idx])]]
        traces.append(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=(i == 0),
            name='Inter-cluster' if i == 0 else None,
            hoverinfo='skip',
        ))

    return traces


# ============================================================================
# Public API
# ============================================================================

def visualize_clusters_html(
    mesh,
    positions: np.ndarray,
    camera_positions: np.ndarray,
    compare_modes: dict,
    original_path_length_mm: float,
    output_html: str,
):
    """Plotly HTML 시각화 (드롭다운 비교 지원).

    Args:
        compare_modes: dict mapping mode_label → {
            'cluster_ids': (N,), 'cluster_order': (K,),
            'path_order': (N,), 'path_length_mm': float, 'num_clusters': int,
        }
    """
    # --- Base traces ---
    base_traces = []
    verts = mesh.vertices
    faces = mesh.faces
    base_traces.append(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='lightgray', opacity=0.25,
        name='Mesh', hoverinfo='skip',
    ))
    n_base = len(base_traces)

    # --- Per-mode traces ---
    mode_names = list(compare_modes.keys())
    all_mode_traces = []
    for mode_label in mode_names:
        m = compare_modes[mode_label]
        traces = _build_cluster_mode_traces(
            camera_positions, m['cluster_ids'], m['cluster_order'],
            m['path_order'], mode_label,
            coacd_parts=m.get('coacd_parts'),
            coacd_ids=m.get('coacd_ids'),
        )
        all_mode_traces.append(traces)

    # --- Assemble figure ---
    fig_traces = list(base_traces)
    mode_trace_ranges = []
    for traces in all_mode_traces:
        start = len(fig_traces)
        fig_traces.extend(traces)
        mode_trace_ranges.append((start, len(fig_traces)))

    fig = go.Figure(data=fig_traces)

    for i, trace in enumerate(fig.data):
        if i < n_base:
            trace.visible = True
        elif mode_trace_ranges[0][0] <= i < mode_trace_ranges[0][1]:
            trace.visible = True
        else:
            trace.visible = False

    if len(mode_names) > 1:
        best_pl = min(compare_modes[m]['path_length_mm'] for m in mode_names)
        buttons = []
        for mode_idx, mode_label in enumerate(mode_names):
            m = compare_modes[mode_label]
            pl_mm = m['path_length_mm']
            K = m['num_clusters']
            diff = ((pl_mm / best_pl) - 1.0) * 100.0
            star = " *" if pl_mm == best_pl else ""
            diff_str = "best" if pl_mm == best_pl else f"+{diff:.1f}%"
            button_label = f"{mode_label}: {pl_mm:.0f}mm ({K} clusters, {diff_str}){star}"

            visibility = [True] * n_base + [False] * (len(fig_traces) - n_base)
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
        title=f'Clustered Viewpoints ({len(positions)} pts) | Original: {original_path_length_mm:.0f}mm',
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
        ),
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    fig.write_html(output_html)
    print(f"  HTML visualization saved to {output_html}")

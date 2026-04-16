"""
Visualize motion trajectory from CSV as Plotly HTML.

Generates two HTML files:
  1. Static trajectory: EE path on mesh with reconfig segments highlighted
  2. Animated trajectory: Slider-based UR5e FK animation

Usage:
    python coverage_tsp/visualize_trajectory.py \
        --csv_path result/motions/bunny/model_inst0.csv \
        --mesh_path data/objects/stanford_bunny/mesh/stanford_bunny.obj \
        --workspace_center 0.45 0.1 0.3
"""

import os
import sys
import csv
import argparse
import numpy as np

import plotly.graph_objects as go
import trimesh
import yourdfpy

# ── UR5e config ────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UR5E_URDF = os.path.join(
    _SCRIPT_DIR, '..', 'curobo', 'src', 'curobo', 'content',
    'assets', 'robot', 'ur_description', 'ur5e.urdf',
)

_LINK_COLORS = [
    'rgb(60,60,60)', 'rgb(80,80,90)', 'rgb(70,120,180)',
    'rgb(70,120,180)', 'rgb(90,90,100)', 'rgb(90,90,100)', 'rgb(50,50,55)',
]

_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# UR5e DH parameters for stick figure FK
_DH_D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
_DH_A = np.array([0, -0.425, -0.3922, 0, 0, 0])
_DH_ALPHA = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

# Reconfig thresholds
_POS_THRESH = 0.2      # meters
_ANGLE_THRESH = 0.5    # radians (z-axis)
_JOINT_THRESH = 1.0    # radians (max joint diff)


# ── CSV parsing ────────────────────────────────────────────────────────────

def load_motion_csv(csv_path):
    """Load motion CSV, return joint_configs (N,6), poses (N,7)."""
    joints = []
    poses = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            jc = []
            for jn in _JOINT_NAMES:
                # CSV header uses 'ur5e-joint_name' format
                for key in row:
                    if jn in key:
                        jc.append(float(row[key]))
                        break
            joints.append(jc)
            poses.append([
                float(row['target-POS_X']), float(row['target-POS_Y']),
                float(row['target-POS_Z']), float(row['target-ROT_X']),
                float(row['target-ROT_Y']), float(row['target-ROT_Z']),
                float(row['target-ROT_W']),
            ])
    return np.array(joints), np.array(poses)


# ── Stick figure FK ────────────────────────────────────────────────────────

def _dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1],
    ])


def _fk_joint_positions(q):
    """Compute 7 joint positions (base + 6 joints) for stick figure."""
    positions = [np.array([0, 0, 0])]  # base
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh_matrix(q[i], _DH_D[i], _DH_A[i], _DH_ALPHA[i])
        positions.append(T[:3, 3].copy())
    return np.array(positions)  # (7, 3)


# ── Reconfig detection ─────────────────────────────────────────────────────

def detect_reconfigs(joints, poses):
    """Return boolean array (N-1,) indicating reconfig between step i and i+1."""
    N = len(joints)
    reconfigs = np.zeros(N - 1, dtype=bool)
    for i in range(N - 1):
        pos_dist = np.linalg.norm(poses[i + 1, :3] - poses[i, :3])
        max_joint_diff = np.max(np.abs(joints[i + 1] - joints[i]))

        # Z-axis angle from quaternion
        def z_axis(q):
            x, y, z, w = q
            return np.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x**2 + y**2)])
        z0 = z_axis(poses[i, 3:])
        z1 = z_axis(poses[i + 1, 3:])
        cos_angle = np.clip(np.dot(z0, z1), -1, 1)
        angle = np.arccos(cos_angle)

        if pos_dist > _POS_THRESH or angle > _ANGLE_THRESH or max_joint_diff > _JOINT_THRESH:
            reconfigs[i] = True
    return reconfigs


# ── Robot mesh loading ─────────────────────────────────────────────────────

def _load_robot(joint_angles):
    """Load UR5e meshes at given joint config. Returns [(verts, faces, name), ...]."""
    urdf_path = os.path.abspath(_UR5E_URDF)
    robot = yourdfpy.URDF.load(
        urdf_path, load_meshes=True, build_scene_graph=True,
        mesh_dir=os.path.dirname(urdf_path),
    )
    cfg = dict(zip(_JOINT_NAMES, joint_angles))
    robot.update_cfg(cfg)

    link_meshes = []
    scene = robot.scene
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if geom is None or not isinstance(geom, trimesh.Trimesh):
            continue
        verts = np.array(geom.vertices, dtype=np.float64)
        ones = np.ones((len(verts), 1))
        verts_h = np.hstack([verts, ones])
        verts_world = (transform @ verts_h.T).T[:, :3]
        faces = np.array(geom.faces)
        link_meshes.append((verts_world, faces, node_name))
    return link_meshes


def _add_robot_traces(fig, joint_angles, opacity=0.7, showlegend=True, visible=True):
    """Add UR5e mesh traces to figure."""
    link_meshes = _load_robot(joint_angles)
    traces = []
    for idx, (v, f, name) in enumerate(link_meshes):
        trace = go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=_LINK_COLORS[idx % len(_LINK_COLORS)],
            opacity=opacity,
            name=f'UR5e: {name}',
            showlegend=(idx == 0 and showlegend),
            legendgroup='ur5e',
            visible=visible,
        )
        traces.append(trace)
        fig.add_trace(trace)
    return traces


# ── Object mesh ────────────────────────────────────────────────────────────

def _add_mesh_trace(fig, mesh_path, workspace_center):
    """Add object mesh at workspace_center."""
    mesh = trimesh.load(mesh_path, process=True, force='mesh')
    centroid = mesh.centroid
    verts = np.array(mesh.vertices, dtype=np.float64)
    verts_centered = verts - centroid
    verts_centered[:, 0] += workspace_center[0]
    verts_centered[:, 1] += workspace_center[1]
    verts_centered[:, 2] += workspace_center[2]
    faces = np.array(mesh.faces)

    fig.add_trace(go.Mesh3d(
        x=verts_centered[:, 0], y=verts_centered[:, 1], z=verts_centered[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='lightblue', opacity=0.3, name='Object mesh',
    ))


# ── Static trajectory visualization ───────────────────────────────────────

def build_static_figure(csv_path, mesh_path, workspace_center, joints, poses, reconfigs):
    """Static trajectory: EE path + reconfig segments + robot at start."""
    fig = go.Figure()

    # Object mesh
    _add_mesh_trace(fig, mesh_path, workspace_center)

    # Robot at initial configuration
    _add_robot_traces(fig, joints[0], opacity=0.5)

    N = len(poses)
    ee_pos = poses[:, :3]

    # Normal segments (green)
    norm_x, norm_y, norm_z = [], [], []
    # Reconfig segments (red)
    rc_x, rc_y, rc_z = [], [], []

    for i in range(N - 1):
        x_seg = [ee_pos[i, 0], ee_pos[i + 1, 0], None]
        y_seg = [ee_pos[i, 1], ee_pos[i + 1, 1], None]
        z_seg = [ee_pos[i, 2], ee_pos[i + 1, 2], None]
        if reconfigs[i]:
            rc_x.extend(x_seg)
            rc_y.extend(y_seg)
            rc_z.extend(z_seg)
        else:
            norm_x.extend(x_seg)
            norm_y.extend(y_seg)
            norm_z.extend(z_seg)

    n_rc = reconfigs.sum()
    fig.add_trace(go.Scatter3d(
        x=norm_x, y=norm_y, z=norm_z, mode='lines',
        line=dict(color='green', width=3),
        name=f'Normal ({N - 1 - n_rc} segments)',
    ))
    fig.add_trace(go.Scatter3d(
        x=rc_x, y=rc_y, z=rc_z, mode='lines',
        line=dict(color='red', width=4),
        name=f'Reconfig ({n_rc} segments)',
    ))

    # Pose markers with step number
    colors = np.arange(N)
    fig.add_trace(go.Scatter3d(
        x=ee_pos[:, 0], y=ee_pos[:, 1], z=ee_pos[:, 2],
        mode='markers',
        marker=dict(size=3, color=colors, colorscale='Viridis',
                    colorbar=dict(title='Step', x=1.05), opacity=0.8),
        text=[f'Step {i}' for i in range(N)],
        hoverinfo='text',
        name='Poses (visit order)',
    ))

    # Start / End markers
    fig.add_trace(go.Scatter3d(
        x=[ee_pos[0, 0]], y=[ee_pos[0, 1]], z=[ee_pos[0, 2]],
        mode='markers', marker=dict(size=8, color='lime', symbol='diamond'),
        name='Start',
    ))
    fig.add_trace(go.Scatter3d(
        x=[ee_pos[-1, 0]], y=[ee_pos[-1, 1]], z=[ee_pos[-1, 2]],
        mode='markers', marker=dict(size=8, color='orange', symbol='square'),
        name='End',
    ))

    basename = os.path.basename(csv_path)
    fig.update_layout(
        title=f'Trajectory: {basename} | {N} poses, {n_rc} reconfigs ({100*n_rc/(N-1):.1f}%)',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                   aspectmode='data'),
        legend=dict(x=0, y=1),
        width=1200, height=800,
    )
    return fig


# ── Animated trajectory visualization ──────────────────────────────────────

def build_animated_figure(csv_path, mesh_path, workspace_center, joints, poses, reconfigs):
    """Animated trajectory: slider-based UR5e stick figure at each step."""
    fig = go.Figure()
    N = len(poses)
    ee_pos = poses[:, :3]

    # Trace 0: Object mesh
    _add_mesh_trace(fig, mesh_path, workspace_center)

    # Trace 1: Full path (dim)
    fig.add_trace(go.Scatter3d(
        x=ee_pos[:, 0], y=ee_pos[:, 1], z=ee_pos[:, 2],
        mode='lines+markers',
        line=dict(color='lightgray', width=2),
        marker=dict(size=2, color='lightgray'),
        name='Full path',
    ))

    # Trace 2: Current EE marker
    fig.add_trace(go.Scatter3d(
        x=[ee_pos[0, 0]], y=[ee_pos[0, 1]], z=[ee_pos[0, 2]],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Current pose',
    ))

    # Trace 3: Normal path so far (green)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode='lines', line=dict(color='green', width=4),
        name='Normal',
    ))

    # Trace 4: Reconfig path so far (red)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode='lines', line=dict(color='red', width=4),
        name='Reconfig',
    ))

    # Build frames — path grows step by step
    print(f"Building {N} animation frames...")
    frames = []
    for step in range(N):
        rc_so_far = int(reconfigs[:max(step, 1)].sum()) if step > 0 else 0

        # Build path segments with reconfig coloring
        norm_x, norm_y, norm_z = [], [], []
        rc_x, rc_y, rc_z = [], [], []
        for i in range(step):
            seg = [ee_pos[i, 0], ee_pos[i+1, 0], None], \
                  [ee_pos[i, 1], ee_pos[i+1, 1], None], \
                  [ee_pos[i, 2], ee_pos[i+1, 2], None]
            if reconfigs[i]:
                rc_x.extend(seg[0]); rc_y.extend(seg[1]); rc_z.extend(seg[2])
            else:
                norm_x.extend(seg[0]); norm_y.extend(seg[1]); norm_z.extend(seg[2])

        frame_data = [
            # Trace 2: current marker
            go.Scatter3d(
                x=[ee_pos[step, 0]], y=[ee_pos[step, 1]], z=[ee_pos[step, 2]],
                mode='markers',
                marker=dict(size=6, color='red'),
            ),
            # Trace 3: normal path so far (green)
            go.Scatter3d(
                x=norm_x, y=norm_y, z=norm_z,
                mode='lines', line=dict(color='green', width=4),
            ),
            # Trace 4: reconfig path so far (red)
            go.Scatter3d(
                x=rc_x, y=rc_y, z=rc_z,
                mode='lines', line=dict(color='red', width=4),
            ),
        ]

        frames.append(go.Frame(
            data=frame_data,
            traces=[2, 3, 4],
            name=str(step),
            layout=go.Layout(title_text=f'Step {step}/{N-1} | Reconfigs so far: {rc_so_far}'),
        ))

    fig.frames = frames

    # Slider
    sliders = [dict(
        active=0,
        currentvalue=dict(prefix='Step: '),
        pad=dict(t=50),
        steps=[
            dict(args=[[str(s)], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                 label=str(s), method='animate')
            for s in range(N)
        ],
    )]

    # Play/Pause buttons
    updatemenus = [dict(
        type='buttons', showactive=False,
        x=0.1, y=0, xanchor='right', yanchor='top',
        pad=dict(t=87, r=10),
        buttons=[
            dict(label='Play', method='animate',
                 args=[None, dict(frame=dict(duration=200, redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label='Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=True),
                                    mode='immediate', transition=dict(duration=0))]),
        ],
    )]

    n_rc = reconfigs.sum()
    basename = os.path.basename(csv_path)
    fig.update_layout(
        title=f'Animation: {basename} | {N} poses, {n_rc} reconfigs',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                   aspectmode='data'),
        sliders=sliders,
        updatemenus=updatemenus,
        width=1200, height=800,
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize motion trajectory')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--workspace_center', type=float, nargs=3,
                        default=[0.45, 0.1, 0.3])
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as CSV)')
    args = parser.parse_args()

    wc = tuple(args.workspace_center)
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.csv_path) or '.'

    # Load CSV
    print(f"Loading: {args.csv_path}")
    joints, poses = load_motion_csv(args.csv_path)
    N = len(joints)
    print(f"  {N} steps, joints shape: {joints.shape}, poses shape: {poses.shape}")

    # Detect reconfigs
    reconfigs = detect_reconfigs(joints, poses)
    n_rc = reconfigs.sum()
    print(f"  Reconfigs: {n_rc}/{N-1} ({100*n_rc/(N-1):.1f}%)")

    basename = os.path.splitext(os.path.basename(args.csv_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    # Static trajectory
    print("Building static trajectory...")
    fig_static = build_static_figure(
        args.csv_path, args.mesh_path, wc, joints, poses, reconfigs)
    static_path = os.path.join(args.output_dir, f'{basename}_trajectory.html')
    fig_static.write_html(static_path)
    print(f"  Saved: {static_path}")

    # Animated trajectory
    print("Building animated trajectory...")
    fig_anim = build_animated_figure(
        args.csv_path, args.mesh_path, wc, joints, poses, reconfigs)
    anim_path = os.path.join(args.output_dir, f'{basename}_animation.html')
    fig_anim.write_html(anim_path)
    print(f"  Saved: {anim_path}")


if __name__ == '__main__':
    main()

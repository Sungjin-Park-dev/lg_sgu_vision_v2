#!/usr/bin/env python3
"""스캔 뷰어 — 촬영한 depth 를 world 로 펼쳐 보고, 복원 메시·정답과 겹쳐 본다 (viser).

점은 복원과 같은 판정(core.reconstruction.segment)으로 나눠 그린다: 물체로 남긴 점(프레임별
색), 씬 장애물이라 지운 점(빨강), ROI 밖이라 지운 점(회색). 물체 점이 정답 메시 표면에 얹히면
(거리 ≈ 0) 내부 파라미터·카메라 pose·depth 규약과 지우기가 모두 맞다는 뜻이다 — 시작할 때
프레임별 거리 표를 찍는다. 정답 메시는 시뮬레이션이라 있는 것이다(core.reconstruction.sim).

reconstruct.py 를 돌렸으면 복원 메시(recon{_tag}.ply)도 토글로 그린다.

Usage:
    uv run scripts/apps/scan_viewer.py --object cylinder_sample
    uv run scripts/apps/scan_viewer.py --object curved_structure --tag n1_0.2_yes
    uv run scripts/apps/scan_viewer.py --object cylinder_sample --no-viser   # 표만
"""

from __future__ import annotations

import argparse
import colorsys
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config, scene_config  # noqa: E402
from core.reconstruction import sim, storage  # noqa: E402
from core.reconstruction.pipeline import FusionParams  # noqa: E402
from core.reconstruction.segment import OBJECT, OBSTACLE, OUTSIDE, classify_pixels  # noqa: E402

OBSTACLE_RGB = (230, 70, 70)       # 씬 장애물이라 지운 점
OUTSIDE_RGB = (130, 130, 130)      # ROI 밖이라 지운 점


def frame_colors(n: int) -> np.ndarray:
    return np.array([[int(255 * c) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 0.95)]
                     for i in range(n)], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True)
    scene_config.add_cli_argument(p)
    p.add_argument("--name", default="default")
    p.add_argument("--tag", default="", help="recon_{tag}.ply 를 그린다")
    p.add_argument("--stride", type=int, default=2, help="depth 픽셀 간격 (점 수 조절)")
    p.add_argument("--roi-only", action="store_true", help="ROI 구 밖의 점은 그리지 않는다")
    p.add_argument("--obstacle-margin-mm", type=float, default=FusionParams().obstacle_margin_mm,
                   help="reconstruct.py 와 같은 값이어야 같은 점이 지워진다")
    p.add_argument("--no-viser", action="store_true", help="표만 출력하고 끝낸다")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8082)
    return p.parse_args()


def main():
    args = parse_args()
    scene_config.apply_cli(args, config)
    config.apply_object_placement(args.object)
    scan = storage.scan_dir(args.object, args.name)
    plan = storage.load_plan(scan)
    spec, roi_c, roi_r = plan.camera, plan.roi_center, plan.roi_radius
    frames = storage.load_frames(scan)
    gt = sim.ground_truth_mesh(args.object)

    margin_m = args.obstacle_margin_mm / 1000.0
    print(f"{'frame':>10} {'object':>8} {'obstacle':>9} {'outside':>8} "
          f"{'med mm':>7} {'p95 mm':>7} {'p99 mm':>7}   (object points → ground truth)")
    layers = {"object": [], "obstacle": [], "outside": []}
    rng = np.random.default_rng(0)
    for fr in frames:
        pts, labels = classify_pixels(fr["depth"], fr["K"], fr["world_T_cam"], spec, roi_c, roi_r,
                                      margin_m, stride=args.stride)
        for key, lab in (("object", OBJECT), ("obstacle", OBSTACLE), ("outside", OUTSIDE)):
            layers[key].append(pts[labels == lab])
        obj = layers["object"][-1]
        sample = obj if len(obj) <= 20000 else obj[rng.choice(len(obj), 20000, replace=False)]
        if len(sample):
            dist_mm = sim.closest_on(gt, sample)[0] * 1000.0
            med, p95, p99 = (float(np.percentile(dist_mm, q)) for q in (50, 95, 99))
        else:
            med = p95 = p99 = float("nan")
        print(f"{fr['name']:>10} {len(obj):8d} {len(layers['obstacle'][-1]):9d} "
              f"{len(layers['outside'][-1]):8d} {med:7.3f} {p95:7.3f} {p99:7.3f}")
    if args.no_viser:
        return

    import trimesh
    import viser
    from scipy.spatial.transform import Rotation

    server = viser.ViserServer(host=args.host, port=args.port)
    gt_handle = server.scene.add_mesh_simple(
        "/ground_truth", vertices=gt.vertices.astype(np.float32), faces=gt.faces.astype(np.int32),
        color=(200, 200, 200), opacity=0.35)
    roi = trimesh.creation.icosphere(subdivisions=2, radius=roi_r)
    server.scene.add_mesh_simple("/roi", vertices=(roi.vertices + roi_c).astype(np.float32),
                                 faces=roi.faces.astype(np.int32), color=(80, 160, 255),
                                 opacity=0.08)
    recon = storage.load_recon(scan, args.tag)
    recon_handle = None
    if recon is not None:
        recon_handle = server.scene.add_mesh_simple(
            "/reconstruction", vertices=recon.vertices.astype(np.float32),
            faces=recon.faces.astype(np.int32), color=(90, 200, 120), opacity=1.0)
        recon_handle.visible = False
    colors = frame_colors(len(frames))
    aspect = spec.width / spec.height
    vfov = np.radians(spec.vfov_deg)

    def cloud(path, pts, rgb, limit=60000):
        if len(pts) > limit:
            pts = pts[rng.choice(len(pts), limit, replace=False)]
        return server.scene.add_point_cloud(path, points=pts.astype(np.float32), colors=rgb,
                                            point_size=0.0015)

    handles = {"object": [], "obstacle": [], "outside": [], "camera": []}
    for i, fr in enumerate(frames):
        base = f"/frames/{fr['name']}"
        handles["object"].append(cloud(f"{base}/object", layers["object"][i], colors[i]))
        handles["obstacle"].append(cloud(f"{base}/obstacle", layers["obstacle"][i], OBSTACLE_RGB))
        handles["outside"].append(
            None if args.roi_only else cloud(f"{base}/outside", layers["outside"][i], OUTSIDE_RGB, 20000))
        T = fr["world_T_cam"]
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        handles["camera"].append(server.scene.add_camera_frustum(
            f"{base}/camera", fov=vfov, aspect=aspect, scale=0.04,
            color=tuple(int(c) for c in colors[i]), wxyz=(w, x, y, z), position=tuple(T[:3, 3])))

    with server.gui.add_folder("Scan"):
        sl = server.gui.add_slider("Frame (−1 = all)", min=-1, max=len(frames) - 1, step=1,
                                   initial_value=-1)
        toggles = {
            "object": server.gui.add_checkbox("Object points (kept)", initial_value=True),
            "obstacle": server.gui.add_checkbox("Removed: known obstacles (red)", initial_value=True),
            "outside": server.gui.add_checkbox("Removed: outside ROI (gray)", initial_value=False,
                                               disabled=args.roi_only),
            "camera": server.gui.add_checkbox("Cameras", initial_value=True),
        }
        cb_gt = server.gui.add_checkbox("Ground truth", initial_value=True)
        cb_rec = server.gui.add_checkbox("Reconstruction", initial_value=False,
                                         disabled=recon_handle is None)

    def refresh(_=None):
        for key, hs in handles.items():
            for i, h in enumerate(hs):
                if h is not None:
                    h.visible = toggles[key].value and (sl.value < 0 or sl.value == i)

    refresh()
    sl.on_update(refresh)
    for cb in toggles.values():
        cb.on_update(refresh)
    if recon_handle is not None:
        cb_rec.on_update(lambda _: setattr(recon_handle, "visible", cb_rec.value))
    cb_gt.on_update(lambda _: setattr(gt_handle, "visible", cb_gt.value))
    print(f"viser: http://localhost:{args.port}  (Ctrl+C to quit)", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

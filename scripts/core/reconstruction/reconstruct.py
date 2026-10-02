#!/usr/bin/env python3
"""캡처한 depth 프레임을 TSDF 로 융합해 물체 메시를 만들고, 정답 메시와 비교한다.

순서:
  1–2. 마스킹(segment.classify_pixels) — 각 depth 픽셀을 world 로 펼쳐, ROI 밖이거나 **씬
     장애물 상자 안**(+ 여유)인 픽셀은 0(=무효)으로 지운다. 셀 기하(테이블·지지대·벽)는 씬 YAML 이 아는 것이고 모르는 건
     물체뿐이라, 아는 것을 지우면 물체만 남는다. (처음엔 RANSAC 으로 바닥 평면을 찾았는데,
     ROI 안 가장 큰 수평면이 지지대가 아니라 원기둥 윗면이라 물체가 통째로 지워졌다.)
  3. TSDF 융합 — 기본은 카메라 pose 를 그대로 믿는다(시뮬레이션이라 정확).
     ``--pose-noise-mm/--pose-noise-deg`` 로 측정 pose 에 오차를 섞어 실물을 흉내 내고,
     ``--icp`` 로 그 pose 를 프레임끼리 맞춰 바로잡는다(registration.py).
  4. 메시 추출 → 가장 큰 덩어리만 → 단순화.
  5. 평가 (정답이 있을 때만):
       정확도  복원 표면 → 정답 표면 거리
       완전성  정답 표면(바닥면 제외) → 복원 표면 거리, 1·2mm 안 비율
       법선    정답 표면점의 법선 ↔ 가장 가까운 복원 면의 법선 각도 — viewpoint 입사각에 직결

출력(world 좌표): data/{object}/scan/{name}/recon{_tag}.ply, recon{_tag}.json

Usage:
    uv run scripts/core/reconstruction/reconstruct.py --object cylinder_sample
    uv run scripts/core/reconstruction/reconstruct.py --object cylinder_sample --voxel-mm 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config, scene_config  # noqa: E402
from core.reconstruction.registration import perturb_poses, pose_errors, refine_poses  # noqa: E402
from core.reconstruction.segment import LABEL_NAMES, OBJECT, classify_pixels  # noqa: E402
from core.reconstruction.sensor import ScanCameraSpec  # noqa: E402
from core.reconstruction.view_scan import closest_on, ground_truth_mesh, load_frames  # noqa: E402

# 정답 표면 중 "아래를 보는 면" — 평면 위에 놓인 물체의 바닥은 어느 카메라도 못 본다. 완전성
# 분모에서 뺀다: 법선이 −z 에서 이 각도 안이면 바닥면.
DOWN_FACING_DEG = 10.0


def surface_samples(mesh, n: int, seed: int = 0):
    """trimesh 메시 표면의 고른 점과 그 면 법선."""
    import trimesh

    pts, fi = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    return pts, mesh.face_normals[fi]


def evaluate(recon, gt, n_samples: int = 50000) -> dict:
    """정확도·완전성·법선 오차. recon/gt 는 trimesh, 같은 world 좌표."""
    rp, _ = surface_samples(recon, n_samples, seed=1)
    acc, _ = closest_on(gt, rp)

    gp, gn = surface_samples(gt, n_samples, seed=2)
    seen = gn[:, 2] > -np.cos(np.radians(DOWN_FACING_DEG))
    gp, gn = gp[seen], gn[seen]
    comp, rn = closest_on(recon, gp)
    cosang = np.abs(np.sum(gn * rn, axis=1))      # 복원 면의 감김 방향이 뒤집혀도 각도는 같게
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    near = comp <= 0.002                            # 법선은 복원이 있는 곳에서만 의미가 있다
    mm = lambda a, q: float(np.percentile(a, q) * 1000.0)   # noqa: E731
    return {
        "accuracy_mm": {"median": mm(acc, 50), "p95": mm(acc, 95), "max": float(acc.max() * 1000)},
        "completeness": {"within_1mm": float((comp <= 0.001).mean()),
                         "within_2mm": float((comp <= 0.002).mean()),
                         "median_mm": mm(comp, 50), "p95_mm": mm(comp, 95)},
        "normal_deg": {"median": float(np.median(ang[near])) if near.any() else None,
                       "p95": float(np.percentile(ang[near], 95)) if near.any() else None},
        "gt_samples_excluding_bottom": int(seen.sum()),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True)
    scene_config.add_cli_argument(p)
    p.add_argument("--name", default="default")
    p.add_argument("--voxel-mm", type=float, default=1.0, help="TSDF 복셀 크기")
    p.add_argument("--trunc-voxels", type=float, default=4.0, help="TSDF truncation = 복셀 × 이 값")
    p.add_argument("--obstacle-margin-mm", type=float, default=1.5,
                   help="씬 장애물 상자를 이만큼 부풀려 그 안의 점을 지운다 (지지대 윗면 등)")
    p.add_argument("--smooth-iters", type=int, default=10, help="Taubin 평활화 반복 수 (0 = 안 함)")
    p.add_argument("--target-faces", type=int, default=20000, help="단순화 목표 면 수 (0 = 안 함)")
    p.add_argument("--pose-noise-mm", type=float, default=0.0,
                   help="측정 pose 에 섞을 프레임별 이동 오차 (RMS, mm) — 시뮬레이션 전용")
    p.add_argument("--pose-noise-deg", type=float, default=0.0,
                   help="측정 pose 에 섞을 프레임별 회전 오차 (RMS, deg)")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--icp", action="store_true", help="융합 전에 ICP 다중 정렬로 pose 를 바로잡는다")
    p.add_argument("--tag", default="", help="출력 이름 recon_{tag}.ply — 실험끼리 덮어쓰지 않게")
    p.add_argument("--no-eval", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    scene_config.apply_cli(args, config)
    config.apply_object_placement(args.object)
    scan = config.DATA_ROOT / args.object / "scan" / args.name
    plan = np.load(scan / "plan.npz")
    spec = ScanCameraSpec.from_dict(json.loads((scan / "plan.json").read_text())["camera"])
    roi_c, roi_r = plan["roi_center"], float(plan["roi_radius"])
    frames = load_frames(scan)
    t0 = time.time()
    stem = f"recon_{args.tag}" if args.tag else "recon"

    # 1. 카메라 pose — 참값(렌더에 쓴 것) → (선택) 오차 섞기 → (선택) ICP 로 바로잡기
    true_poses = np.stack([fr["world_T_cam"] for fr in frames])
    poses = true_poses
    pose_report = {}

    def report_pose(label, used):
        e = pose_errors(true_poses, used, roi_c)
        (ts, tr), (rs, rr) = e["total"], e["relative"]
        pose_report[label] = {
            "total_shift_mm": {"median": float(np.median(ts)), "max": float(ts.max())},
            "relative_shift_mm": {"median": float(np.median(rs)), "max": float(rs.max())},
            "relative_rot_deg": {"median": float(np.median(rr)), "max": float(rr.max())},
            "common_shift_mm": e["common"][0], "common_rot_deg": e["common"][1]}
        print(f"[pose] {label:9s} at ROI centre — total median {np.median(ts):.2f} / max {ts.max():.2f} mm | "
              f"frame-to-frame median {np.median(rs):.2f} / max {rs.max():.2f} mm, "
              f"{np.median(rr):.3f}° | common offset {e['common'][0]:.2f} mm, {e['common'][1]:.3f}°")

    if args.pose_noise_mm or args.pose_noise_deg:
        poses = perturb_poses(true_poses, args.pose_noise_mm, args.pose_noise_deg, args.noise_seed)
        report_pose("noisy", poses)
    if args.icp:
        poses, pose_report["icp"] = refine_poses(frames, poses, spec, roi_c, roi_r)
        report_pose("after ICP", poses)

    # 2–3. 마스킹 + TSDF
    import open3d as o3d

    voxel = args.voxel_mm / 1000.0
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * args.trunc_voxels,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    counts = np.zeros(len(LABEL_NAMES), dtype=np.int64)
    for fr, pose in zip(frames, poses):
        depth = fr["depth"].astype(np.float64).copy()
        _, labels = classify_pixels(depth, fr["K"], pose, spec, roi_c, roi_r,
                                    args.obstacle_margin_mm / 1000.0)
        depth[labels != OBJECT] = 0.0               # Open3D 는 0 을 무효 depth 로 본다
        counts += np.bincount(labels.ravel(), minlength=len(LABEL_NAMES))
        h, w = depth.shape
        K = fr["K"]
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        d_img = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8)), d_img,
            depth_scale=1.0, depth_trunc=spec.max_range_m, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(pose))
    total = counts.sum()
    print("[recon] pixels: " + ", ".join(f"{LABEL_NAMES[k]} {counts[k] / total * 100:.1f}%"
                                         for k in range(len(LABEL_NAMES))))
    print(f"[recon] fused {len(frames)} frames, {counts[OBJECT]} object pixels "
          f"(voxel {args.voxel_mm} mm, trunc {args.voxel_mm * args.trunc_voxels:.1f} mm)")

    # 4. 메시
    mesh = vol.extract_triangle_mesh()
    tri_cluster, cluster_n, _ = mesh.cluster_connected_triangles()
    tri_cluster = np.asarray(tri_cluster)
    biggest = int(np.argmax(np.asarray(cluster_n)))
    mesh.remove_triangles_by_mask(tri_cluster != biggest)
    mesh.remove_unreferenced_vertices()
    n_raw = len(mesh.triangles)
    if args.smooth_iters:
        # marching cubes 삼각형은 계단처럼 깔려 면 법선이 흔들린다(복셀을 줄여도 안 줄고 오히려
        # 는다). Taubin 은 수축 없이 고주파만 깎는다.
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iters)
    if args.target_faces and n_raw > args.target_faces:
        mesh = mesh.simplify_quadric_decimation(args.target_faces)
    mesh.compute_vertex_normals()
    print(f"[recon] mesh: {n_raw} -> {len(mesh.triangles)} triangles "
          f"({len(cluster_n)} components, kept the largest) in {time.time() - t0:.1f}s")

    import trimesh

    recon = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
    recon.export(str(scan / f"{stem}.ply"))
    lo, hi = recon.bounds
    report = {
        "voxel_mm": args.voxel_mm, "smooth_iters": args.smooth_iters, "trunc_mm": args.voxel_mm * args.trunc_voxels,
        "obstacle_margin_mm": args.obstacle_margin_mm,
        "pose_noise": {"trans_mm": args.pose_noise_mm, "rot_deg": args.pose_noise_deg,
                       "seed": args.noise_seed},
        "icp": bool(args.icp), "pose_errors": pose_report,
        "faces": int(len(recon.faces)), "watertight": bool(recon.is_watertight),
        "extents_mm": ((hi - lo) * 1000).round(2).tolist(),
    }

    # 5. 평가
    if not args.no_eval:
        gt = ground_truth_mesh(args.object)
        report["eval"] = evaluate(recon, gt)
        report["gt_extents_mm"] = (gt.extents * 1000).round(2).tolist()
        e = report["eval"]
        print(f"[recon] extents {report['extents_mm']} mm  (ground truth {report['gt_extents_mm']})")
        print(f"[recon] accuracy   (recon→gt): median {e['accuracy_mm']['median']:.3f}  "
              f"p95 {e['accuracy_mm']['p95']:.3f}  max {e['accuracy_mm']['max']:.3f} mm")
        print(f"[recon] completeness (gt→recon, bottom excluded): "
              f"{e['completeness']['within_1mm'] * 100:.1f}% ≤1mm, "
              f"{e['completeness']['within_2mm'] * 100:.1f}% ≤2mm, p95 {e['completeness']['p95_mm']:.2f} mm")
        print(f"[recon] normals: median {e['normal_deg']['median']:.2f}°  p95 {e['normal_deg']['p95']:.2f}°")
    (scan / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[recon] -> {(scan / f'{stem}.ply').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""촬영 프레임 → 물체 메시(world 좌표) → data/{object}/scan/{name}/recon{_tag}.ply/.json.

복원 자체는 pipeline.reconstruct 다. 이 스크립트는 거기에 시뮬레이션 실험을 두른다(sim.py):
  --pose-noise-mm/--pose-noise-deg  측정 pose 에 프레임별 오차를 섞어 실물(FK + hand-eye)을 흉내 낸다
  --icp                             그 pose 를 프레임끼리 맞춰 바로잡는다
  (기본)                            정답 메시와 비교해 정확도·완전성·법선을 남긴다 (--no-eval 로 끔)

Usage:
    uv run scripts/core/reconstruction/reconstruct.py --object cylinder_sample
    uv run scripts/core/reconstruction/reconstruct.py --object curved_structure \\
        --pose-noise-mm 1 --pose-noise-deg 0.2 --icp --tag n1_icp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config, scene_config  # noqa: E402
from core.reconstruction import sim, storage  # noqa: E402
from core.reconstruction.pipeline import FusionParams, reconstruct  # noqa: E402


def parse_args():
    d = FusionParams()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True)
    scene_config.add_cli_argument(p)
    p.add_argument("--name", default="default")
    p.add_argument("--voxel-mm", type=float, default=d.voxel_mm, help="TSDF 복셀 크기")
    p.add_argument("--trunc-voxels", type=float, default=d.trunc_voxels,
                   help="TSDF truncation = 복셀 × 이 값")
    p.add_argument("--obstacle-margin-mm", type=float, default=d.obstacle_margin_mm,
                   help="씬 장애물 상자를 이만큼 부풀려 그 안의 점을 지운다 (지지대 윗면 등)")
    p.add_argument("--smooth-iters", type=int, default=d.smooth_iters,
                   help="Taubin 평활화 반복 수 (0 = 안 함)")
    p.add_argument("--target-faces", type=int, default=d.target_faces, help="단순화 목표 면 수 (0 = 안 함)")
    p.add_argument("--pose-noise-mm", type=float, default=0.0,
                   help="측정 pose 에 섞을 프레임별 이동 오차 (RMS, mm) — 시뮬레이션 전용")
    p.add_argument("--pose-noise-deg", type=float, default=0.0,
                   help="측정 pose 에 섞을 프레임별 회전 오차 (RMS, deg)")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--icp", action="store_true", help="융합 전에 ICP 다중 정렬로 pose 를 바로잡는다")
    p.add_argument("--tag", default="", help="출력 이름 recon_{tag}.ply — 실험끼리 덮어쓰지 않게")
    p.add_argument("--no-eval", action="store_true")
    return p.parse_args()


def print_pose(label: str, summary: dict) -> None:
    t, r = summary["total_shift_mm"], summary["relative_shift_mm"]
    print(f"[pose] {label:9s} at ROI centre — total median {t['median']:.2f} / max {t['max']:.2f} mm | "
          f"frame-to-frame median {r['median']:.2f} / max {r['max']:.2f} mm, "
          f"{summary['relative_rot_deg']['median']:.3f}° | common offset "
          f"{summary['common_shift_mm']:.2f} mm, {summary['common_rot_deg']:.3f}°")


def print_eval(e: dict, extents, gt_extents) -> None:
    print(f"[recon] extents {extents} mm  (ground truth {gt_extents})")
    print(f"[recon] accuracy   (recon→gt): median {e['accuracy_mm']['median']:.3f}  "
          f"p95 {e['accuracy_mm']['p95']:.3f}  max {e['accuracy_mm']['max']:.3f} mm")
    print(f"[recon] completeness (gt→recon, bottom excluded): "
          f"{e['completeness']['within_1mm'] * 100:.1f}% ≤1mm, "
          f"{e['completeness']['within_2mm'] * 100:.1f}% ≤2mm, p95 {e['completeness']['p95_mm']:.2f} mm")
    print(f"[recon] normals: median {e['normal_deg']['median']:.2f}°  p95 {e['normal_deg']['p95']:.2f}°")


def main():
    args = parse_args()
    scene_config.apply_cli(args, config)
    config.apply_object_placement(args.object)
    scan = storage.scan_dir(args.object, args.name)
    plan = storage.load_plan(scan)
    frames = storage.load_frames(scan)
    params = FusionParams(voxel_mm=args.voxel_mm, trunc_voxels=args.trunc_voxels,
                          obstacle_margin_mm=args.obstacle_margin_mm,
                          smooth_iters=args.smooth_iters, target_faces=args.target_faces)

    # 측정 pose — 시뮬레이션은 렌더에 쓴 참 pose 가 저장돼 있어, 원하면 오차를 섞는다.
    true_poses = np.stack([fr["world_T_cam"] for fr in frames])
    poses = true_poses
    pose_report = {}
    if args.pose_noise_mm or args.pose_noise_deg:
        poses = sim.perturb_poses(true_poses, args.pose_noise_mm, args.pose_noise_deg, args.noise_seed)
        pose_report["noisy"] = sim.pose_error_summary(true_poses, poses, plan.roi_center)
        print_pose("noisy", pose_report["noisy"])

    result = reconstruct(frames, plan, params, poses=poses, icp=args.icp)
    if args.icp:
        pose_report["after ICP"] = sim.pose_error_summary(true_poses, result.poses, plan.roi_center)
        print_pose("after ICP", pose_report["after ICP"])

    recon = result.mesh
    lo, hi = recon.bounds
    report = {
        **params.to_dict(), "trunc_mm": params.voxel_mm * params.trunc_voxels,
        "pose_noise": {"trans_mm": args.pose_noise_mm, "rot_deg": args.pose_noise_deg,
                       "seed": args.noise_seed},
        "icp": bool(args.icp), "pose_errors": pose_report, "stats": result.stats,
        "faces": int(len(recon.faces)), "watertight": bool(recon.is_watertight),
        "extents_mm": ((hi - lo) * 1000).round(2).tolist(),
    }
    if not args.no_eval:
        gt = sim.ground_truth_mesh(args.object)
        report["eval"] = sim.evaluate(recon, gt)
        report["gt_extents_mm"] = (gt.extents * 1000).round(2).tolist()
        print_eval(report["eval"], report["extents_mm"], report["gt_extents_mm"])
    path = storage.save_recon(scan, recon, report, tag=args.tag)
    print(f"[recon] -> {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

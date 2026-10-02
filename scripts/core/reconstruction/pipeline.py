"""복원 파이프라인 — 촬영 프레임 → 물체 메시. 앱·CLI 가 부르는 진입점은 ``reconstruct``.

  1. (선택) ICP 다중 정렬로 카메라 pose 를 바로잡는다 (registration.refine_poses)
  2. 마스킹 — 각 depth 픽셀을 world 로 펼쳐, ROI 밖이거나 **씬 장애물 상자 안**(+ 여유)인
     픽셀은 0(=무효)으로 지운다(segment.classify_pixels). 셀 기하(테이블·지지대·벽)는 씬 YAML 이
     아는 것이고 모르는 건 물체뿐이라, 아는 것을 지우면 물체만 남는다. (처음엔 RANSAC 으로 바닥
     평면을 찾았는데, ROI 안 가장 큰 수평면이 지지대가 아니라 원기둥 윗면이라 물체가 통째로
     지워졌다.)
  3. TSDF 융합 (Open3D ScalableTSDFVolume)
  4. 메시 추출 → 가장 큰 덩어리만 → Taubin 평활화 → 단순화

메시는 world 좌표다. 정답과 비교하거나 오차를 일부러 섞는 일은 여기 없다(sim.py).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from core.reconstruction.registration import refine_poses
from core.reconstruction.segment import LABEL_NAMES, OBJECT, classify_pixels
from core.reconstruction.storage import ScanPlan


@dataclass(frozen=True)
class FusionParams:
    voxel_mm: float = 1.0
    trunc_voxels: float = 4.0            # TSDF truncation = 복셀 × 이 값
    obstacle_margin_mm: float = 1.5      # 씬 장애물 상자를 이만큼 부풀려 그 안의 점을 지운다
    # marching cubes 삼각형은 계단처럼 깔려 면 법선이 흔들린다(복셀을 줄여도 안 줄고 오히려 는다 —
    # 0.5mm 복셀에서 법선 중앙값 3.7° → 5.7°). Taubin 은 수축 없이 고주파만 깎는다(→ 2.1°).
    smooth_iters: int = 10
    target_faces: int = 20000            # 0 = 단순화 안 함

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReconResult:
    mesh: object                        # trimesh.Trimesh, world 좌표
    poses: np.ndarray                   # (N, 4, 4) 융합에 쓴 카메라 pose
    stats: dict = field(default_factory=dict)


def fuse_tsdf(frames, poses, plan: ScanPlan, params: FusionParams):
    """마스킹한 depth 를 TSDF 에 쌓는다 → (Open3D TSDF 볼륨, 픽셀 판정 개수 (len(LABEL_NAMES),))."""
    import open3d as o3d

    spec = plan.camera
    voxel = params.voxel_mm / 1000.0
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * params.trunc_voxels,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    counts = np.zeros(len(LABEL_NAMES), dtype=np.int64)
    for fr, pose in zip(frames, poses):
        depth = fr["depth"].astype(np.float64).copy()
        _, labels = classify_pixels(depth, fr["K"], pose, spec, plan.roi_center, plan.roi_radius,
                                    params.obstacle_margin_mm / 1000.0)
        depth[labels != OBJECT] = 0.0               # Open3D 는 0 을 무효 depth 로 본다
        counts += np.bincount(labels.ravel(), minlength=len(LABEL_NAMES))
        h, w = depth.shape
        K = fr["K"]
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8)),
            o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=1.0, depth_trunc=spec.max_range_m, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(pose))
    return vol, counts


def extract_mesh(vol, params: FusionParams):
    """TSDF → 정리된 trimesh. (메시, {raw_faces, components})."""
    import trimesh

    mesh = vol.extract_triangle_mesh()
    tri_cluster, cluster_n, _ = mesh.cluster_connected_triangles()
    tri_cluster = np.asarray(tri_cluster)
    if not len(cluster_n):
        raise ValueError("TSDF 에서 메시가 하나도 안 나왔다 — 물체 픽셀이 0 인지 확인")
    mesh.remove_triangles_by_mask(tri_cluster != int(np.argmax(np.asarray(cluster_n))))
    mesh.remove_unreferenced_vertices()
    n_raw = len(mesh.triangles)
    if params.smooth_iters:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=params.smooth_iters)
    if params.target_faces and n_raw > params.target_faces:
        mesh = mesh.simplify_quadric_decimation(params.target_faces)
    out = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
    return out, {"raw_faces": int(n_raw), "components": int(len(cluster_n))}


def reconstruct(frames, plan: ScanPlan, params: FusionParams = FusionParams(), *,
                poses: np.ndarray | None = None, icp: bool = False, log=print) -> ReconResult:
    """프레임 → 물체 메시(world 좌표).

    poses: 융합에 쓸 카메라 pose. 기본은 프레임에 저장된 world_T_cam(촬영 순간의 측정값).
    icp: 융합 전에 ICP 다중 정렬로 pose 를 바로잡는다.
    """
    t0 = time.time()
    if poses is None:
        poses = np.stack([fr["world_T_cam"] for fr in frames])
    stats: dict = {}
    if icp:
        poses, stats["icp"] = refine_poses(frames, poses, plan.camera, plan.roi_center,
                                           plan.roi_radius, log=log)

    vol, counts = fuse_tsdf(frames, poses, plan, params)
    total = max(int(counts.sum()), 1)
    stats["pixels"] = {LABEL_NAMES[k]: int(counts[k]) for k in range(len(LABEL_NAMES))}
    log("[recon] pixels: " + ", ".join(f"{LABEL_NAMES[k]} {counts[k] / total * 100:.1f}%"
                                       for k in range(len(LABEL_NAMES))))
    log(f"[recon] fused {len(frames)} frames, {counts[OBJECT]} object pixels "
        f"(voxel {params.voxel_mm} mm, trunc {params.voxel_mm * params.trunc_voxels:.1f} mm)")

    mesh, mstats = extract_mesh(vol, params)
    stats.update(mstats)
    stats["seconds"] = round(time.time() - t0, 1)
    log(f"[recon] mesh: {mstats['raw_faces']} -> {len(mesh.faces)} triangles "
        f"({mstats['components']} components, kept the largest) in {stats['seconds']}s")
    return ReconResult(mesh=mesh, poses=poses, stats=stats)

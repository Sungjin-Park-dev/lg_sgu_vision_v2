"""시뮬레이션 전용 — 정답 CAD 가 있을 때만 쓰는 것들. 실물 파이프라인은 이 모듈을 import 하지 않는다.

  world_T_object / ground_truth_mesh   씬 배치로 놓은 정답 source.obj (world 좌표)
  roi_from_ground_truth                정답 bbox 를 감싸는 ROI 구 — 실물에선 사용자가 준다
  perturb_poses / pose_errors          측정 pose 오차를 섞고, 남은 오차를 공통/프레임 간으로 나눠 잰다
  closest_on / evaluate                복원 메시를 정답과 비교 (정확도·완전성·법선)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from common import config
from common.math_utils import quaternion_to_rotation_matrix

# 정답 표면 중 "아래를 보는 면" — 평면 위에 놓인 물체의 바닥은 어느 카메라도 못 본다. 완전성
# 분모에서 뺀다: 법선이 −z 에서 이 각도 안이면 바닥면.
DOWN_FACING_DEG = 10.0


def world_T_object() -> np.ndarray:
    """씬 배치(config.TARGET_OBJECT)의 4×4 — apply_object_placement 뒤에 부른다."""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(np.asarray(config.TARGET_OBJECT["rotation"], dtype=np.float64))
    T[:3, 3] = np.asarray(config.TARGET_OBJECT["position"], dtype=np.float64)
    return T


def ground_truth_mesh(object_name: str):
    """정답 source.obj 를 씬 배치로 world 에 놓은 trimesh."""
    import trimesh

    mesh = trimesh.load(str(config.get_mesh_path(object_name, mesh_type="source")), force="mesh")
    mesh.apply_transform(world_T_object())
    return mesh


def roi_from_ground_truth(object_name: str, margin_m: float = 0.02):
    """정답 메시의 배치 bbox 를 감싸는 구 → (center (3,), radius)."""
    lo, hi = ground_truth_mesh(object_name).bounds
    return 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo)) + margin_m


# ---------------------------------------------------------------------------
# 카메라 pose 오차
# ---------------------------------------------------------------------------

def perturb_poses(poses: np.ndarray, trans_mm: float, rot_deg: float, seed: int = 0) -> np.ndarray:
    """프레임마다 독립인 강체 오차를 섞은 pose. (N, 4, 4) → (N, 4, 4).

    이동은 등방 가우시안으로, 축마다 σ = trans_mm/√3 이라 크기의 RMS 가 trans_mm 이다. 회전도
    같은 식으로 rot_deg 가 RMS 각도다. 오차는 카메라 프레임에서 걸린다(카메라를 그 자리에서
    살짝 밀고 돌린 것).
    """
    rng = np.random.default_rng(seed)
    n = len(poses)
    t = rng.normal(0.0, trans_mm / np.sqrt(3.0), size=(n, 3)) / 1000.0
    r = rng.normal(0.0, np.radians(rot_deg) / np.sqrt(3.0), size=(n, 3))
    E = np.tile(np.eye(4), (n, 1, 1))
    E[:, :3, :3] = Rotation.from_rotvec(r).as_matrix()
    E[:, :3, 3] = t
    return poses @ E


def mean_transform(Ts: np.ndarray) -> np.ndarray:
    """강체 변환들의 평균 — 회전은 쿼터니언 평균, 이동은 산술 평균."""
    M = np.eye(4)
    M[:3, :3] = Rotation.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
    M[:3, 3] = Ts[:, :3, 3].mean(axis=0)
    return M


def _shift_rot(D: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shift = np.linalg.norm((D @ p)[..., :3] - p[:3], axis=-1) * 1000.0
    cos = (np.trace(D[..., :3, :3], axis1=-2, axis2=-1) - 1.0) / 2.0
    return shift, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def pose_errors(true_poses: np.ndarray, used_poses: np.ndarray, point) -> dict:
    """pose 오차를 두 갈래로 나눠 잰다 — 점 ``point``(ROI 중심)의 밀림 mm 와 회전 deg.

    카메라 원점의 이동만 보면 회전 오차가 숨는다. 물체가 있는 자리가 그 pose 로 얼마나 밀려
    보이는지가 복원에 실제로 들어가는 오차다.

      total     프레임별 전체 오차
      common    모든 프레임에 공통인 부분(평균 변환) — 복원을 통째로 옮길 뿐 모양은 안 망친다
      relative  공통 부분을 뺀 프레임끼리의 불일치 — 표면을 겹겹이 어긋나게 만드는 쪽
    """
    p = np.append(np.asarray(point, dtype=np.float64), 1.0)
    # used 로 world 에 놓은 점 = used @ inv(true) @ p_true — 참 자리에서 얼마나 옮겨졌나
    D = used_poses @ np.linalg.inv(true_poses)
    M = mean_transform(D)
    common = _shift_rot(M, p)
    return {"total": _shift_rot(D, p), "common": (float(common[0]), float(common[1])),
            "relative": _shift_rot(np.linalg.inv(M) @ D, p)}


def pose_error_summary(true_poses, used_poses, point) -> dict:
    """pose_errors 를 JSON 으로 남길 요약 수치로."""
    e = pose_errors(true_poses, used_poses, point)
    (ts, _), (rs, rr) = e["total"], e["relative"]
    return {"total_shift_mm": {"median": float(np.median(ts)), "max": float(ts.max())},
            "relative_shift_mm": {"median": float(np.median(rs)), "max": float(rs.max())},
            "relative_rot_deg": {"median": float(np.median(rr)), "max": float(rr.max())},
            "common_shift_mm": e["common"][0], "common_rot_deg": e["common"][1]}


# ---------------------------------------------------------------------------
# 정답 대비 평가
# ---------------------------------------------------------------------------

def closest_on(mesh, query):
    """정확한 최근접 거리(m)와 그 면의 법선 — Open3D RaycastingScene."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
    tm.triangle.indices = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32))
    scene.add_triangles(tm)
    res = scene.compute_closest_points(o3d.core.Tensor(np.asarray(query, dtype=np.float32)))
    nearest = res["points"].numpy().astype(np.float64)
    dist = np.linalg.norm(nearest - query, axis=1)
    return dist, np.asarray(mesh.face_normals)[res["primitive_ids"].numpy()]


def surface_samples(mesh, n: int, seed: int = 0):
    """trimesh 메시 표면의 고른 점과 그 면 법선."""
    import trimesh

    pts, fi = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    return pts, mesh.face_normals[fi]


def evaluate(recon, gt, n_samples: int = 50000) -> dict:
    """정확도·완전성·법선 오차. recon/gt 는 trimesh, 같은 world 좌표.

      정확도  복원 표면 → 정답 표면 거리
      완전성  정답 표면(바닥면 제외) → 복원 표면 거리, 1·2mm 안 비율
      법선    정답 표면점의 법선 ↔ 가장 가까운 복원 면의 법선 각도 — viewpoint 입사각에 직결
    """
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

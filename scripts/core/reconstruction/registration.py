"""카메라 pose 를 ICP 다중 정렬로 바로잡는다 — ``refine_poses``.

실물에서는 프레임마다 카메라 pose 를 로봇 FK + hand-eye 보정으로 얻는다. 거기엔 수백 µm ~ 수 mm
오차가 섞인다. pose 가 틀리면 TSDF 가 같은 표면을 여러 겹으로 어긋나게 쌓는다. 측정 pose 를
초기값으로 프레임 쌍별 ICP → 사전정보 붙은 포즈 그래프 최소제곱을 푼다. (오차를 일부러 섞는
쪽은 시뮬레이션 전용이라 sim.perturb_poses 에 있다.)

refine_poses 가 바로잡는 것은 **프레임끼리의 불일치**다. 모든 프레임에 똑같이 들어간 오차
(예: hand-eye 보정의 고정 편차)는 프레임끼리 비교해서는 보이지 않는다 — 그건 복원 전체를
통째로 옮길 뿐이라 모양은 멀쩡하고, 물체가 놓인 위치만 틀린다.

측정 pose 는 **사전정보(prior)** 로 들어간다: 보정량 ξ_i 가 0 근처에 있으라는 약한 제약. 이게
없으면 데이터가 말하지 않는 방향으로 해가 흘러간다 — 원기둥은 축 둘레로 돌려도 모양이 같아서
그 회전을 ICP 가 못 잡는데, 사전정보 없는 Open3D 포즈 그래프는 거기서 207mm·39° 로 발산했다.
사전정보는 전체 기준(gauge)도 정해 준다: 프레임끼리만 비교하면 전체를 통째로 옮겨도 똑같이
맞는데, 그 공통 성분은 측정 pose 들의 (가중) 평균에 머문다 — 오차가 평균 0 인 무작위라면
가장 좋은 추정이다.

풀이: 오차가 1° 미만으로 작아 작은 각도 근사 ξ = [ω, t] (T ≈ I + [ω]×, t) 로 선형화된다. 간선
(j→i, ICP 가 j 를 i 에 맞추는 T_ij) 마다 ξ_j − ξ_i ≈ vee(T_ij), 노드마다 ξ_i ≈ 0. 정보행렬
가중 최소제곱을 풀고, 보정된 pose 로 ICP 를 다시 돌려 몇 번 되풀이한다. 회전·이동이
world 원점(로봇 베이스, 물체에서 ~0.8m)에서 강하게 얽히지 않도록 ROI 중심을 원점으로 옮겨 푼다.

두 가지를 조심한다(처음 판은 이 둘 때문에 보정이 실제 오차보다 커지며 발산했다):
  * 정보행렬은 **점-면** 으로 직접 만든다. Open3D 의 get_information_matrix 는 점-점이라 면을 따라
    미끄러지는 방향까지 안다고 주장하는데, 점-면 ICP 는 그 방향을 잡지 않는다.
  * 잘못 맞춰진 쌍이 섞인다. 간선 잔차에 Cauchy 가중(IRLS)을 걸어 큰 잔차의 간선을 눌러 둔다.

Open3D ICP 는 멀티스레드라 같은 입력에서도 실행마다 결과가 µm 단위로 다르다(curved_structure
1mm/0.2° 에서 프레임 간 오차 중앙값 0.2754 vs 0.2760mm).

알려진 한계: 원기둥처럼 대칭인 물체는 대칭 방향(축 회전)의 정보가 "거의 0" 이지 0 은 아니라,
ICP 가 그 방향으로 내놓는 헛값이 사전정보를 조금 이긴다(1mm/0.2° 오차에서 축 회전이 0.19° →
0.88°). 대칭이라 모양은 안 변한다. 작은 고유값을 잘라 보았지만 curved_structure 가 오히려
나빠져(프레임 간 최대 0.42 → 0.90mm) 넣지 않았다.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

from core.reconstruction.segment import OBJECT, classify_pixels


def _frame_cloud(frame, pose, spec, roi_c, roi_r, margin_m, voxel_m, stride, obstacles):
    """물체 점(ROI 중심 원점 좌표) + 법선 — 측정 pose 로 펼친다."""
    import open3d as o3d

    pts, labels = classify_pixels(frame["depth"], frame["K"], pose, spec, roi_c, roi_r,
                                  margin_m, stride=stride, obstacles=obstacles)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[labels == OBJECT] - roi_c))
    pcd = pcd.voxel_down_sample(voxel_m)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4.0 * voxel_m, max_nn=30))
    pcd.orient_normals_towards_camera_location(pose[:3, 3] - roi_c)
    return pcd


def _vee(T: np.ndarray) -> np.ndarray:
    """작은 강체 변환 → [ω, t] (Open3D 정보행렬과 같은 순서: 회전 먼저)."""
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def _exp(xi: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(xi[:3]).as_matrix()
    T[:3, 3] = xi[3:]
    return T


def _point_to_plane_info(src, tgt, T, max_corr_m: float) -> tuple[np.ndarray, int]:
    """점-면 ICP 의 정보행렬 (6×6, 단위 분산) — 대응점마다 J = [q×n, n] 한 줄. (정보행렬, 대응점 수).

    법선 방향 잔차만 재므로 면을 따라 미끄러지는 방향은 정보가 0 이다. 원기둥의 축 회전처럼
    데이터가 말하지 않는 방향이 그대로 드러나, 거기서는 사전정보가 해를 붙든다.
    """
    from scipy.spatial import cKDTree

    sp = np.asarray(src.points) @ T[:3, :3].T + T[:3, 3]
    tp, tn = np.asarray(tgt.points), np.asarray(tgt.normals)
    dist, idx = cKDTree(tp).query(sp, distance_upper_bound=max_corr_m)
    ok = np.isfinite(dist)
    q, nrm = tp[idx[ok]], tn[idx[ok]]
    J = np.hstack([np.cross(q, nrm), nrm])
    return J.T @ J, int(ok.sum())


def refine_poses(frames, poses, spec, roi_center, roi_radius, *,
                 prior_trans_mm: float = 2.0, prior_rot_deg: float = 0.3, icp_sigma_mm: float = 1.0,
                 obstacle_margin_m: float = 0.006, voxel_m: float = 0.0015, stride: int = 2,
                 max_pair_angle_deg: float = 60.0, min_fitness: float = 0.2,
                 corr_schedule_m=(0.008, 0.004, 0.002), rounds: int = 3,
                 robust_mm: float = 0.5, irls_iters: int = 10, obstacles=None, log=print) -> tuple[np.ndarray, dict]:
    """측정 pose 를 ICP 다중 정렬로 바로잡는다 → (보정된 pose (N,4,4), 진단).

    prior_*: 측정 pose 를 얼마나 믿나(로봇 FK + hand-eye 오차의 대략적 크기). 데이터가 잡아 주는
        방향에서는 ICP 가 압도하고, 대칭이라 못 잡는 방향에서만 이 값이 해를 붙든다.
    icp_sigma_mm: 대응점 하나의 잔차 크기 가정. 정보행렬을 단위 분산(1 m²)으로 만들었으므로
        이걸로 나눠 사전정보와 같은 척도로 맞춘다.
    robust_mm: Cauchy 가중의 척도. 간선 잔차(대응점의 법선 방향 RMS 밀림)가 이만큼이면 가중이 절반.
    obstacle_margin_m: 복원 때(1.5mm)보다 넉넉하다 — 측정 pose 가 수 mm 틀린 상태에서 지지대
        점이 섞이면 ICP 가 물체가 아니라 지지대에 맞춘다. 정렬에는 물체 점만 있으면 된다.
    """
    import open3d as o3d

    t0 = time.time()
    n = len(frames)
    C = np.eye(4)
    C[:3, 3] = roi_center                              # ROI 중심 원점 좌표 → world
    prior = np.diag([1.0 / np.radians(prior_rot_deg) ** 2] * 3 + [1.0 / (prior_trans_mm / 1000.0) ** 2] * 3)
    icp_scale = 1.0 / (icp_sigma_mm / 1000.0) ** 2
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    cos_max = np.cos(np.radians(max_pair_angle_deg))

    xi = np.zeros((n, 6))                              # 측정 pose 대비 누적 보정 (ROI 좌표)
    diag = {"rounds": []}
    for rnd in range(rounds):
        current = C @ np.stack([_exp(x) for x in xi]) @ np.linalg.inv(C) @ poses
        clouds = [_frame_cloud(f, p, spec, roi_center, roi_radius, obstacle_margin_m, voxel_m,
                               stride, obstacles)
                  for f, p in zip(frames, current)]
        dirs = roi_center - current[:, :3, 3]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        edges = []                                     # (i, j, z, info)
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dirs[i] @ dirs[j] < cos_max or not len(clouds[i].points) or not len(clouds[j].points):
                    continue
                n_pairs += 1
                T = np.eye(4)
                for corr in corr_schedule_m:           # 거칠게 → 곱게
                    res = o3d.pipelines.registration.registration_icp(
                        clouds[j], clouds[i], corr, T, estimation,
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
                    T = res.transformation
                if res.fitness < min_fitness:
                    continue
                info_unit, n_corr = _point_to_plane_info(clouds[j], clouds[i], T, corr_schedule_m[-1])
                edges.append((i, j, _vee(T), info_unit, max(n_corr, 1)))   # δ_j − δ_i ≈ z
        n_used = len(edges)

        weights = np.ones(n_used)
        for _ in range(irls_iters):
            H = np.zeros((6 * n, 6 * n))
            b = np.zeros(6 * n)
            for i in range(n):                         # 사전정보: 누적 보정 ξ_i + δ_i ≈ 0
                H[6 * i:6 * i + 6, 6 * i:6 * i + 6] += prior
                b[6 * i:6 * i + 6] -= prior @ xi[i]
            for w, (i, j, z, info_unit, _) in zip(weights, edges):
                si, sj = slice(6 * i, 6 * i + 6), slice(6 * j, 6 * j + 6)
                wi = w * icp_scale * info_unit
                H[si, si] += wi
                H[sj, sj] += wi
                H[si, sj] -= wi
                H[sj, si] -= wi
                b[sj] += wi @ z
                b[si] -= wi @ z
            delta = np.linalg.solve(H, b).reshape(n, 6)
            # 간선 잔차 = 대응점의 법선 방향 RMS 밀림. 정보가 있는 방향만 센다 — 원기둥 축 회전처럼
            # 데이터가 말하지 않는 방향의 차이는 그 간선의 잘못이 아니다.
            resid_mm = np.array([np.sqrt(max(r @ info_unit @ r, 0.0) / nc) * 1000.0
                                 for (i, j, z, info_unit, nc) in edges
                                 for r in [delta[j] - delta[i] - z]])
            new_w = 1.0 / (1.0 + (resid_mm / robust_mm) ** 2)
            if np.allclose(new_w, weights, atol=1e-3):
                break
            weights = new_w
        xi += delta
        step_mm = float(np.abs(delta[:, 3:]).max() * 1000.0)
        n_down = int((weights < 0.5).sum())
        diag["rounds"].append({"pairs_tried": n_pairs, "pairs_used": n_used, "downweighted": n_down,
                               "max_step_mm": step_mm})
        log(f"[icp] round {rnd + 1}: {n_used}/{n_pairs} pairs (≥{min_fitness:.0%} overlap), "
            f"{n_down} down-weighted as outliers, "
            f"~{int(np.median([len(c.points) for c in clouds]))} pts/frame, max step {step_mm:.3f} mm")
        if step_mm < 0.01:
            break

    refined = C @ np.stack([_exp(x) for x in xi]) @ np.linalg.inv(C) @ poses
    diag["seconds"] = round(time.time() - t0, 1)
    log(f"[icp] done in {diag['seconds']}s")
    return refined, diag

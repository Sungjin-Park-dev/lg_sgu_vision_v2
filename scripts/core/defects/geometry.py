"""스크래치를 표면에 붙이고, 스튜디오 좌표로 옮기고, 카메라별로 보이는지 센다.

좌표계 규약: 스크래치(``scratches.json``)는 언제나 ``source.obj`` 좌표계(미터)다 — Isaac 의
``source.usd`` 와 같은 좌표계라 결함의 정답 위치로 그대로 쓸 수 있다. 스튜디오는 ``.stp`` 를
열면 메시를 **고른 부품의 바닥 중심**으로 옮기므로(``core/viewpoint/mesh.py``) 좌표가 다를 수
있다. 실측으로 차이는 평행이동뿐이었다(sample 만 +9.1, 0, −62.1 mm, 회전 동일). 그래서 화면과
가시성 계산 때만 ``FrameAlignment`` 로 옮기고, 그 이동이 맞는지 표면 거리로 확인한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from core.viewpoint import visibility


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def scratch_frame(s: dict):
    """(중심, 길이축, 폭축, 법선) — 길이축은 법선에 직교하게 다시 맞춘다."""
    centre = np.asarray(s["center"], dtype=np.float64)
    normal = _unit(s["normal"])
    along = np.asarray(s["direction"], dtype=np.float64)
    along = _unit(along - normal * float(normal @ along))
    return centre, along, np.cross(normal, along), normal


def project_onto_surface(surface: trimesh.Trimesh, s: dict, along_m, across_m=None,
                         lift_m: float = 0.0):
    """접평면 위 점 ``c + a·길이축 (+ b·폭축)`` 을 법선 반대로 쏴서 표면에 붙인다.

    Blender 단계가 데칼 UV 를 만드는 투영, 그리고 배치 때 "끝이 허공에 뜨는가" 를 보는
    광선과 같은 방향이다 — 그래서 화면의 선이 실제 흠집 자리와 어긋나지 않는다. 광선은
    표면에서 ``max(1mm, L/2)`` 위에서 출발한다: 곡면에서는 접평면 끝이 표면보다 떠 있기 때문이다.
    빗나가거나 너무 먼 곳에 맞으면 가장 가까운 표면점으로 떨어진다.

    Returns: (points (n,3) + 법선×lift, face normals (n,3), hit mask (n,))
    """
    centre, along, across, normal = scratch_frame(s)
    a = np.atleast_1d(np.asarray(along_m, dtype=np.float64))
    b = np.zeros_like(a) if across_m is None else np.atleast_1d(np.asarray(across_m, float))
    tangent = centre + a[:, None] * along + b[:, None] * across
    length_m = float(s["length_mm"]) / 1000.0
    origins = tangent + normal * max(1e-3, length_m / 2.0)
    directions = np.repeat(-normal[None], len(tangent), axis=0)

    points = tangent.copy()
    normals = np.repeat(normal[None], len(tangent), axis=0)
    hit = np.zeros(len(tangent), dtype=bool)
    locs, rays, tris = surface.ray.intersects_location(origins, directions, multiple_hits=False)
    for loc, ray, tri in zip(locs, rays, tris):
        if np.linalg.norm(loc - tangent[ray]) <= length_m:
            points[ray], normals[ray], hit[ray] = loc, surface.face_normals[tri], True
    if (~hit).any():
        closest, _, tris = trimesh.proximity.closest_point(surface, tangent[~hit])
        points[~hit] = closest
        normals[~hit] = surface.face_normals[tris]
    return points + normals * lift_m, normals, hit


def scratch_polyline(surface: trimesh.Trimesh, s: dict, n: int = 9,
                     lift_m: float = 3e-4) -> np.ndarray:
    """화면용 — 표면을 따라가는 선. 살짝 띄워 메시에 묻히지 않게 한다."""
    half = float(s["length_mm"]) / 2000.0
    points, _, _ = project_onto_surface(surface, s, np.linspace(-half, half, n), lift_m=lift_m)
    return points


def scratch_samples(surface: trimesh.Trimesh, s: dict, k: int = 5):
    """가시성용 — 흠집 길이를 따라 k 점과 그 법선. 띄우지 않는다(표면점 그 자체)."""
    half = float(s["length_mm"]) / 2000.0
    points, normals, _ = project_onto_surface(surface, s, np.linspace(-half, half, k))
    return points, normals


def decal_patch(surface: trimesh.Trimesh, s: dict, half_len_m: float, half_wid_m: float,
                span_m: float, nu: int = 24, nv: int = 6, lift_m: float = 1.5e-4):
    """흠집 자리의 작은 격자 메시와 UV — 화면에서 실제 흠집 모양을 입힐 바탕.

    메시의 원래 면을 쓰지 않고 격자를 새로 만든다: sample 의 벽처럼 240mm 삼각형 하나에
    텍스처를 입히면 UV 가 창 밖으로 크게 뻗는다. UV 는 Blender 단계와 같은 투영식
    ``u = a/span + 0.5, v = b/span + 0.5`` 라 텍스처(한가운데 흠집, 긴 축 +u)와 그대로 맞는다.
    광선이 빗나간 꼭짓점이 낀 칸은 버린다(모서리 밖으로 삐져나간 부분).

    Returns: (vertices (n,3), faces (m,3), uv (n,2))
    """
    a = np.linspace(-half_len_m, half_len_m, nu + 1)
    b = np.linspace(-half_wid_m, half_wid_m, nv + 1)
    grid_a, grid_b = np.meshgrid(a, b, indexing="ij")
    points, _, hit = project_onto_surface(surface, s, grid_a.ravel(), grid_b.ravel(),
                                          lift_m=lift_m)
    uv = np.c_[grid_a.ravel() / span_m + 0.5, grid_b.ravel() / span_m + 0.5]
    faces = []
    for i in range(nu):
        for j in range(nv):
            v00, v10 = i * (nv + 1) + j, (i + 1) * (nv + 1) + j
            quad = (v00, v10, v10 + 1, v00 + 1)
            if hit[list(quad)].all():
                faces.extend([(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])])
    return points, np.asarray(faces, dtype=np.int64).reshape(-1, 3), uv


def shade_normal_map(rgb: np.ndarray, alpha: np.ndarray,
                     light=(0.0, 0.87, 0.5)) -> np.ndarray:
    """탄젠트 노멀맵 → 비스듬한 조명으로 한 번 칠한 RGBA. 화면(viser)은 노멀맵을 못 그린다.

    빛은 흠집을 **가로질러** 들어온다(Blender 미리보기와 같은 방향) — 홈의 한쪽 벽은 밝고
    반대쪽은 어두워 흠집으로 읽힌다. 평평한 곳의 밝기를 기준으로 차이만 키워 칠하고,
    투명도는 흠집 자신의 알파다(평평한 여백은 완전히 투명).
    """
    n = rgb.astype(np.float64) / 255.0 * 2.0 - 1.0
    l = np.asarray(light, dtype=np.float64)
    l = l / np.linalg.norm(l)
    relief = n @ l - l[2]                         # 평평한 면 = 0
    grey = np.clip(0.30 + 1.6 * relief, 0.03, 0.95)
    out = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = np.rint(grey * 255.0)[..., None].astype(np.uint8)
    out[..., 3] = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# source.obj 좌표 ↔ 스튜디오 좌표
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameAlignment:
    """스튜디오 좌표 = source.obj 좌표 + ``offset_m``. 맞는지 표면 거리로 확인한 결과를 같이 든다."""

    offset_m: np.ndarray
    median_mm: float
    p99_mm: float
    ok: bool
    note: str

    def to_scene(self, points) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) + self.offset_m

    def to_source(self, points) -> np.ndarray:
        """화면에서 찍은 점을 스크래치 좌표로 — 클릭 배치(다음 단계)가 쓴다."""
        return np.asarray(points, dtype=np.float64) - self.offset_m


IDENTITY = FrameAlignment(np.zeros(3), 0.0, 0.0, True, "same frame")


def _bottom_centre(mesh: trimesh.Trimesh) -> np.ndarray:
    lo, hi = mesh.bounds
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])


def align_source_to_scene(source_full: trimesh.Trimesh, scene_full: trimesh.Trimesh, *,
                          source_target: trimesh.Trimesh | None = None,
                          scene_target: trimesh.Trimesh | None = None,
                          tol_mm: float = 1.0, n_probe: int = 400,
                          seed: int = 0) -> FrameAlignment:
    """두 좌표계 사이의 평행이동을 찾고, 그게 맞는지 확인한다.

    후보는 둘이다 — 어셈블리 전체 bbox 중심의 차, 그리고 (있으면) 검사 부품 바닥 중심의 차.
    뒤의 것이 ``load_meshes`` 가 실제로 하는 일(고른 부품 바닥 중심으로 옮기기)이다. source 표면
    점을 옮겨 scene 표면까지 거리를 재고, p99 가 가장 작은 후보를 고른다. 회전이 다르거나
    (align 이 모호/none) 부품 구성이 달라 이동만으로 안 맞으면 ``ok=False`` — 틀린 자리에
    그리느니 그리지 않는다.
    """
    candidates = [scene_full.bounds.mean(axis=0) - source_full.bounds.mean(axis=0)]
    if source_target is not None and scene_target is not None:
        candidates.append(_bottom_centre(scene_target) - _bottom_centre(source_target))
    probes, _ = trimesh.sample.sample_surface(source_full, n_probe, seed=seed)

    best = None
    for offset in candidates:
        distance = trimesh.proximity.closest_point(scene_full, probes + offset)[1] * 1000.0
        score = (float(np.percentile(distance, 99)), float(np.median(distance)), offset)
        if best is None or score[0] < best[0]:
            best = score
    p99, median, offset = best
    ok = p99 <= tol_mm and median <= 0.5 * tol_mm
    note = (f"offset {np.round(offset * 1000, 1).tolist()} mm · p99 {p99:.2f} mm" if ok else
            f"좌표계 불일치 — 이동만으로 안 맞는다 (p99 {p99:.1f} mm)")
    return FrameAlignment(np.asarray(offset, dtype=np.float64), median, p99, ok, note)


# ---------------------------------------------------------------------------
# 스크래치별 가시성
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScratchVisibility:
    n_full: int                 # 샘플점 k 개를 **한 대가 모두** 보는 카메라 수
    n_partial: int              # 일부만 보는 카메라 수
    points_seen: int            # k 점 중 어느 카메라든 본 점의 수
    k: int
    views_full: np.ndarray      # n_full 카메라의 인덱스
    state: str                  # "seen" | "split" | "missed"


def scratch_visibility(samples, view_positions, view_normals, occluder,
                       spec: visibility.SensorSpec, fov_mm=None,
                       frames: visibility.ViewFrames | None = None) -> list[ScratchVisibility]:
    """스크래치마다 몇 대가 온전히/부분적으로 보는가.

    ``samples`` 는 스크래치별 ``(points (k,3), normals (k,3))`` — **이미 스튜디오 좌표**다.
    판정은 ``visibility.sees_all_pairs`` 한 번으로 끝난다(모든 스크래치의 점을 쌓아서).

    상태:
      * seen   — 한 장의 사진에 흠집 전체가 들어오는 카메라가 있다
      * split  — 점들은 누군가 보지만 한 장에 다 들어오지는 않는다(FOV 보다 긴 흠집 등)
      * missed — 어느 카메라도 흠집의 어떤 점도 못 본다
    """
    if not samples:
        return []
    k = len(samples[0][0])
    points = np.concatenate([np.asarray(p, dtype=np.float64) for p, _ in samples])
    normals = np.concatenate([np.asarray(n, dtype=np.float64) for _, n in samples])
    seen = visibility.sees_all_pairs(view_positions, view_normals, points, normals,
                                     occluder, spec, fov_mm=fov_mm, frames=frames)
    seen = seen.reshape(len(np.asarray(view_positions).reshape(-1, 3)), len(samples), k)

    out = []
    for i in range(len(samples)):
        per_view = seen[:, i, :]
        full = per_view.all(axis=1)
        partial = per_view.any(axis=1) & ~full
        union = per_view.any(axis=0)
        state = "seen" if full.any() else ("split" if union.any() else "missed")
        out.append(ScratchVisibility(int(full.sum()), int(partial.sum()), int(union.sum()), k,
                                     np.flatnonzero(full), state))
    return out

"""Mesh and material loading for viewpoint generation."""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from common import config

# STEP(B-rep)은 삼각형이 아니라 곡면 정의라 테셀레이션을 거쳐야 한다. trimesh 가 cascadio
# (OpenCASCADE 포장)로 처리한다 — 없으면 로드 시점에 안내와 함께 실패한다.
STEP_SUFFIXES = {".stp", ".step"}
# cascadio 기본값은 0.01(파일 단위=mm)이라 curved_structure 가 22,817 삼각형이 된다.
# 0.1mm 면 2,611 개로 기존 source.obj(3,022)와 같은 급이다 — 비교 가능한 기본값.
DEFAULT_STEP_TOL_LINEAR = 0.1

# 자세 정렬 방식. STEP 은 CAD 좌표계를 그대로 들고 오는데 어느 축이 '위' 인지는 파일마다
# 다르다(이 리포: curved/square/cylinder 는 Y-up, sample_step 은 Z-up).
ALIGN_AUTO, ALIGN_Z_UP, ALIGN_Y_UP, ALIGN_NONE = "auto", "z-up", "y-up", "none"
ALIGN_MODES = (ALIGN_AUTO, ALIGN_Z_UP, ALIGN_Y_UP, ALIGN_NONE)
# auto 의 1·2등 점수 차가 이보다 작으면 "사실상 동점" 이라 믿을 수 없다고 경고한다.
ALIGN_AMBIGUOUS_MARGIN = 0.02


def load_source_parts(mesh_path, tol_linear: Optional[float] = None,
                      tol_angular: Optional[float] = None) -> dict:
    """메시 파일 → {이름: geometry}. STEP 은 솔리드 이름, OBJ 는 재질 이름이 키가 된다.

    어셈블리 STEP 은 부품마다 geometry 가 따로 온다(sample_step: SAMPLE / SAMPLE_BRACKET).
    검사 대상만 고르려면 이 이름이 필요하다 — OBJ 의 재질 필터에 대응하는 STEP 쪽 선택자다.
    """
    mesh_path = Path(mesh_path)
    kwargs = {}
    if mesh_path.suffix.lower() in STEP_SUFFIXES:
        try:
            import cascadio  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise ImportError(
                f"STEP 을 읽으려면 cascadio 가 필요하다 (uv pip install cascadio): {mesh_path}"
            ) from exc
        kwargs["tol_linear"] = float(
            DEFAULT_STEP_TOL_LINEAR if tol_linear is None else tol_linear)
        if tol_angular is not None:
            kwargs["tol_angular"] = float(tol_angular)
    loaded = trimesh.load(str(mesh_path), **kwargs)
    if not isinstance(loaded, trimesh.Scene):
        return {mesh_path.stem: loaded}
    # ⚠️ ``scene.geometry`` 는 부품을 **자기 로컬 좌표**로 준다 — 어셈블리 배치는
    # ``scene.graph`` 에 따로 있다. 그걸 안 씌우면 부품들이 제자리를 벗어나 흩어진다
    # (sample_step: 부품이 브래킷에서 떨어져 그려지던 원인). 단일 재질 OBJ 처럼 변환이
    # 단위행렬이면 아무 일도 일어나지 않으므로 항상 씌워도 안전하다.
    placed = {}
    for node in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node]
        part = loaded.geometry[geometry_name].copy()
        part.apply_transform(transform)
        key = geometry_name if geometry_name not in placed else f"{geometry_name}:{node}"
        placed[key] = part
    return placed or dict(loaded.geometry)


def load_source_geometry(mesh_path, tol_linear: Optional[float] = None,
                         tol_angular: Optional[float] = None) -> List[trimesh.Trimesh]:
    """메시 파일 → 부분(geometry) 목록. OBJ 는 재질별로, STEP 은 테셀레이션 결과로 나뉜다.

    STEP 은 ``tol_linear`` (파일 단위, 우리 파일은 mm)로 밀도가 정해진다 — 곡면을 몇 mm 오차로
    근사할지다. 삼각형 개수가 이 값 하나로 수십 배 움직이므로 밖에서 정할 수 있어야 한다.
    """
    return list(load_source_parts(mesh_path, tol_linear, tol_angular).values())


def direction_profile(mesh: trimesh.Trimesh) -> np.ndarray:
    """±X/±Y/±Z 를 향하는 면적 6개 — 물체의 '자세 지문'."""
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                     [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    normals, areas = mesh.face_normals, mesh.area_faces
    return np.array([areas[(normals @ d) > 0.9].sum() for d in dirs])


def _axis_rotations() -> List[np.ndarray]:
    """축정렬 회전 24개(행렬식 +1)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                R[row, col] = sign
            if np.isclose(np.linalg.det(R), 1.0):
                out.append(R)
    return out


def up_axis_transform(mode: str) -> np.ndarray:
    """명시적 자세 규약 → 4x4. CAD 파일이 어느 축을 '위' 로 쓰는지 사람이 아는 경우."""
    T = np.eye(4)
    if mode == ALIGN_Y_UP:      # CAD +Y 를 +Z 로 (이 리포의 대부분 STEP)
        T[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    elif mode not in (ALIGN_Z_UP, ALIGN_NONE):
        raise ValueError(f"unknown align mode: {mode}")
    return T                     # z-up / none 은 회전 없음


def axis_align_transform(mesh: trimesh.Trimesh,
                         reference: trimesh.Trimesh) -> Tuple[np.ndarray, float]:
    """mesh 를 reference 와 같은 자세로 돌리는 4x4 (+ 일치도 0~1).

    STEP 은 CAD 좌표계 그대로라 우리 규약(방향은 메시에 베이크)과 축이 다르다 — 세 부품 모두
    y/z 가 바뀌어 있다. bbox 만 비교하면 위아래가 뒤집힌 회전을 구별하지 못하므로(180° 뒤집어도
    bbox 는 같다) **방향별 면적 분포**로 고른다. 물체가 완전 대칭이 아닌 한 유일하게 정해진다.
    """
    ref = direction_profile(reference)
    ref = ref / max(np.linalg.norm(ref), 1e-12)
    scored = []
    for R in _axis_rotations():
        candidate = mesh.copy()
        T = np.eye(4)
        T[:3, :3] = R
        candidate.apply_transform(T)
        profile = direction_profile(candidate)
        scored.append((float(profile @ ref / max(np.linalg.norm(profile), 1e-12)), T))
    scored.sort(key=lambda item: -item[0])
    best_score, best_T = scored[0]
    margin = best_score - scored[1][0] if len(scored) > 1 else 1.0
    if margin < ALIGN_AMBIGUOUS_MARGIN:
        # 지문이 변별력을 잃었다는 신호다. sample 처럼 STEP(어셈블리)과 source.obj 가 아예
        # 다른 형상이면 여기 걸린다 — 그대로 쓰면 물체가 뒤집힌 채 놓인다.
        print(f"  NOTE: 자세 정렬 1·2등 차가 작다 ({best_score:.4f} vs "
              f"{scored[1][0]:.4f}) — 화면에서 자세를 확인하고, 틀렸으면 align 을 "
              f"y-up/z-up 으로 못박으세요")
    return best_T, best_score


def bottom_center_transform(mesh: trimesh.Trimesh) -> np.ndarray:
    """bbox 바닥 중심을 원점으로 보내는 4x4 — setup/prepare_object_mesh 와 같은 규약."""
    lo, hi = mesh.bounds
    T = np.eye(4)
    T[:3, 3] = -np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]])
    return T


def material_rgb_of(geometry: trimesh.Trimesh) -> Tuple[int, int, int] | None:
    """geometry 의 대표 재질 색 (R,G,B) 0~255. 색을 못 찾으면 None.

    ``visual.material`` (OBJ+MTL → SimpleMaterial, glTF → PBRMaterial) 을 먼저 보고,
    재질이 없는 메시(정점색만 있는 경우)는 ``visual`` 자체의 main_color 로 떨어진다.
    """
    for source in (getattr(geometry.visual, "material", None), geometry.visual):
        color = getattr(source, "main_color", None)
        if color is not None:
            rgba = trimesh.visual.color.to_rgba(color)
            return (int(rgba[0]), int(rgba[1]), int(rgba[2]))
    return None


def split_by_material(loaded) -> List[trimesh.Trimesh]:
    """trimesh 가 재질별로 갈라 놓은 geometry 목록. 단일 메시면 길이 1."""
    if isinstance(loaded, trimesh.Scene):
        return list(loaded.geometry.values())
    return [loaded]


def color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)
                                - np.asarray(b, dtype=np.float64)))


# ============================================================================
# Path & PCA Utilities
# ============================================================================

def load_meshes(object_name, material_rgb=None, color_tolerance=5.0, *,
                mesh_path=None, part_name=None, align=ALIGN_AUTO, tol_linear=None):
    """소스 메시 로드 + (선택) 부품·재질 필터로 타깃 메시 추출.

    Args:
        mesh_path: 기본 ``data/{object}/mesh/source.obj`` 대신 읽을 파일. ``.stp``/``.step``
            은 cascadio 로 테셀레이션한다.
        part_name: 어셈블리에서 고를 부품(솔리드) 이름. STEP 은 부품마다 geometry 가 따로
            오므로(sample_step: SAMPLE / SAMPLE_BRACKET) 검사 대상만 남길 수 있다.
            **자세·원점 정렬은 고른 부품 기준으로** 잡는다 — 지그까지 포함해 맞추면
            검사 대상이 원점에서 벗어난다.
        align: 자세 규약. ``auto`` 는 source.obj 와 방향별 면적 분포를 맞춘다(모호하면
            경고). CAD 파일이 어느 축을 위로 쓰는지 알면 ``z-up``/``y-up`` 으로 못박고,
            ``none`` 이면 CAD 좌표 그대로 둔다.
        tol_linear: STEP 테셀레이션 허용 오차(파일 단위, 우리 파일은 mm).

    Returns: (full_mesh, target_mesh, input_path)
    """
    canonical = Path(config.get_mesh_path(object_name, mesh_type="source"))
    path = Path(mesh_path) if mesh_path else canonical
    input_path = str(path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input mesh not found: {input_path}")
    if align is True:
        align = ALIGN_AUTO
    elif align is False:
        align = ALIGN_NONE

    print("Loading mesh...")
    named = load_source_parts(path, tol_linear=tol_linear)
    selected = named
    if part_name:
        chosen = {k: v for k, v in named.items() if k == part_name}
        if not chosen:
            chosen = {k: v for k, v in named.items() if k.lower() == str(part_name).lower()}
        if not chosen:
            raise ValueError(
                f"part '{part_name}' not found in {path.name} — available: {sorted(named)}")
        selected = chosen
    parts = [part.copy() for part in named.values()]
    keys = list(named)
    selected_idx = [keys.index(k) for k in selected]
    print(f"  Loaded: {sum(len(p.faces) for p in parts):,} triangles in {len(parts)} "
          f"geometry group{'s' if len(parts) != 1 else ''} from {path.name}"
          + (f" — sampling '{list(selected)[0]}'" if part_name else ""))

    is_alternate = path.resolve() != canonical.resolve()
    if is_alternate and align != ALIGN_NONE:
        # 자세·원점은 **검사 대상 부품** 기준으로 잡는다(지그까지 넣어 맞추면 대상이
        # 원점에서 밀린다). 변환 자체는 어셈블리 전체에 똑같이 씌워 상대 배치를 지킨다.
        anchor = trimesh.util.concatenate([parts[i] for i in selected_idx]) \
            if len(selected_idx) > 1 else parts[selected_idx[0]].copy()
        if align == ALIGN_AUTO:
            if not canonical.exists():
                raise ValueError(
                    f"align='auto' 는 기준 메시({canonical.name})가 필요하다 — "
                    f"align 을 z-up/y-up/none 중에서 고르세요")
            reference = trimesh.load(str(canonical), force="mesh")
            transform, score = axis_align_transform(anchor, reference)
            note = f"auto (match {score:.3f} vs {canonical.name})"
        else:
            transform, note = up_axis_transform(align), f"{align} (fixed)"
        anchor.apply_transform(transform)
        transform = bottom_center_transform(anchor) @ transform
        for part in parts:
            part.apply_transform(transform)
        print(f"  Aligned: {note}, bottom-centered on the sampled part")

    # full_mesh 는 **어셈블리 전체**다 — 화면에 지그가 같이 보이고, 가시성 필터가 지그에
    # 가려지는 viewpoint 를 잡아낼 수 있다(부품만 넘기면 그걸 놓친다).
    mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    target_parts = [parts[i] for i in selected_idx]
    print(f"  Extents (mm): {np.round(mesh.extents * 1000, 1)}")
    print()

    if material_rgb:
        target_rgb = tuple(int(v) for v in material_rgb.split(','))
        print("Matching material...")
        matched = []
        for part in target_parts:
            rgb = material_rgb_of(part)
            distance = color_distance(rgb, target_rgb) if rgb is not None else float('inf')
            hit = distance <= color_tolerance
            print(f"    - RGB{rgb}: {len(part.faces):,} faces, "
                  f"distance {distance:.1f} → {'MATCH' if hit else 'skip'}")
            if hit:
                matched.append(part)
        if not matched:
            available = [material_rgb_of(part) for part in target_parts]
            hint = ("\nSTEP/CAD 파일에는 재질 구분이 없다 — material 필터를 비우고 "
                    "Part 로 고르세요." if path.suffix.lower() in STEP_SUFFIXES else "")
            raise ValueError(
                f"No materials matched RGB{target_rgb} within tolerance {color_tolerance}\n"
                f"Available material colors: {available}{hint}")
        target_mesh = trimesh.util.concatenate(matched) if len(matched) > 1 else matched[0]
        pct = len(target_mesh.faces) / len(mesh.faces) * 100
        print(f"  Target: {len(target_mesh.faces):,} / {len(mesh.faces):,} "
              f"triangles ({pct:.1f}%)")
    else:
        print("Using entire mesh (no material filter)...")
        target_mesh = trimesh.util.concatenate(target_parts) \
            if len(target_parts) > 1 else target_parts[0]
        print(f"  Triangles: {len(target_mesh.faces):,}")

    print(f"  Surface area: {target_mesh.area:.6f} m2")
    print()
    return mesh, target_mesh, input_path


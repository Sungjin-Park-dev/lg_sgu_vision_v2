#!/usr/bin/env python3
"""Interactive viewpoint studio with viser.

Two ways to put viewpoints on screen, both object-centric:

  * **Generate** — pick an object, tune the stage parameters, and regenerate
    in-process via the ``viewpoint/cli.py`` seam (``load_meshes`` /
    ``prepare_viewpoints`` / ``build_local_delaunay_adjacency``). The panel follows
    the four stages, in the order they run:

      1. **Candidates** — two samplers. **Surface FPS** scatters points over the
         triangles; **CAD faces** walks each B-rep face of a STEP source, lays out
         camera frames along arc length in (u,v), keeps what falls inside the
         trimming boundary, and takes positions/normals analytically (no
         tessellation error). Both then pass the same bottom/interior/occlusion
         filters (``finalize_viewpoints``).
      2. **Selection** — keep them all, or let greedy set cover pick a minimal
         subset that still meets the coverage target.
      3. **Verify** — count how much of each face is actually covered.
      4. **Solver graph** — build the local-tangent Delaunay graph on whatever
         survived. This stage cannot change the points, only the edges.

    Stages 2 and 3 ask the same question ("does this camera inspect this bit of
    surface?") and get it from the same place, ``viewpoint/visibility.py``, so the
    selected set and the reported coverage cannot disagree.
  * **Saved viewpoints** — load a previously saved ``viewpoints*.h5``.

A viewpoint file carries two layers: **geometry** (positions/normals + camera
spec) and the **local-tangent Delaunay graph** (edges only). It carries no visit
order — GLNS solves the order jointly with the IK configuration, reading only
positions/normals/edges/WD. Clustering and lawnmower ordering belonged to the
plan_trajectory era and were removed on 2026-08-26.

Rendered elements: translucent mesh, surface points, camera positions, and the
graph edges — all coloured by connected component, each toggled independently
under **Display**. One line at the bottom reports the graph the next stage
actually consumes: edge count, component count, isolated points, and the edge
count GLNS will really solve on (**Solver graph (hops)**).

Which faces get sampled is tuned under **Faces**: the material RGB filter, the
bottom-face filter (angle from world −z), the hollow-object interior filter, and
the ray-cast occlusion filter.
The defaults come from the per-object tables in ``config`` — the fields just make
them visible and overridable per run. Found parameters can be persisted with
**Save** for the GLNS solve step.

Usage:
    uv run scripts/apps/viewpoint_studio.py --object sample
    uv run scripts/apps/viewpoint_studio.py --viewpoints data/sample/viewpoint/124/viewpoints.h5
"""

from __future__ import annotations

import argparse
import colorsys
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import trimesh
import viser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> scripts/
from common import config, scene_config
from core.viewpoint import (
    DEFAULT_DELAUNAY_DISTANCE_FACTOR,
    DEFAULT_STEP_TOL_LINEAR,
    DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
    DEFAULT_DELAUNAY_NEIGHBORS,
    ViewpointGenParams,
    build_local_delaunay_adjacency,
    components_from_edges,
    expand_edges_by_hops,
    finalize_viewpoints,
    load_meshes,
    load_source_parts,
    load_viewpoints_hdf5,
    prepare_viewpoints,
    save_viewpoints_hdf5,
    subset_viewpoints,
)
from core.viewpoint import brep, select, visibility

MESH_RGB = (180, 180, 180)
SURFACE_RGB = (255, 255, 255)
# 재질 보기의 불투명도. 단색 보기(0.25)보다 조금 올린다 — 색이 실려 있어 더 옅으면 안 읽힌다.
MATERIAL_ALPHA = 0.35

# 하한이 0 인 이유: CAD faces 는 프레임을 사각형으로 놓으므로 겹침 없이도 면이 타일링된다
# (원판 근사이던 시절에는 r ≥ s·√2/2, 즉 29.3% 겹침이 있어야 모서리가 덮였다).
OVERLAP_MIN_PCT = 0
OVERLAP_MAX_PCT = 90
FOV_MIN_MM = 5.0
FOV_MAX_MM = 500.0
WD_MAX_MM = 800.0
# WD 하한은 물리 제약(검사면이 카메라 끝보다 앞에 있어야 한다)에서 온다. optical_frame 이
# 렌즈 앞면에 있어 config 의 값은 0 이고, 그 경계(= 검사면이 렌즈 앞면과 정확히 겹침)를
# working_distance_error 가 실패로 보므로 입력칸 하한은 1mm 안쪽으로 올린다.
WD_MIN_MM = float(int(config.CAMERA_MIN_WORKING_DISTANCE_MM) + 1)
# Saved-viewpoints 드롭다운의 두 특수 항목. GENERATED 는 "만들었지만 아직 디스크에 없다" 는
# 상태를 드롭다운이 스스로 말하게 하려고 둔다 — 예전에는 Generate 든 Save 든 (none) 이라
# 화면에 점이 132개 떠 있는데 드롭다운은 아무것도 없다고 주장했다.
NONE_LABEL = "(none)"
GENERATED_LABEL = "(generated · unsaved)"
IDLE_HINT = "Pick an object, then **Generate** — or load a saved viewpoint set."

# GLNS 가 이 그래프 위에 얹는 확장(--delaunay-expand-hops). 성분 수는 이걸로 안 바뀐다
# (N-hop 안에 있다는 건 이미 경로가 있다는 뜻이라 같은 성분이다) — 바뀌는 것은 간선 수,
# 즉 GLNS 가 순서를 고를 자유도다. 그래서 여기서는 "GLNS 가 실제로 몇 개의 간선 위에서
# 푸는가" 를 보여준다.
DEFAULT_GLNS_HOPS = 2
MAX_GLNS_HOPS = 4

# 샘플러 두 가지. 이름은 '무엇 위에 점을 뿌리나' 를 말한다 — 삼각형이냐 CAD 곡면이냐.
SAMPLER_FPS = "Surface FPS (mesh)"
SAMPLER_BREP = "CAD faces (STEP)"
# 선택 단계 표기. 기본은 '전부' = 지금까지의 동작.
SELECTION_LABELS = {"전부 사용": select.SELECTION_ALL,
                    "Greedy set cover": select.SELECTION_GREEDY}
STEP_SUFFIXES = (".stp", ".step")
# 어셈블리에서 부품을 안 고른 상태. 파일 전체를 하나로 다룬다.
PART_ALL = "(all)"

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def default_overlap_pct() -> int:
    pct = config.CAMERA_OVERLAP_RATIO * 100.0
    return int(round(_clamp(pct, OVERLAP_MIN_PCT, OVERLAP_MAX_PCT)))


def fov_spacing_mm(overlap_pct: float,
                   fov_w_mm: float = None,
                   fov_h_mm: float = None) -> tuple[float, float, float]:
    """Return (row, col, isotropic_surface) spacing in mm from FOV and overlap."""
    if fov_w_mm is None:
        fov_w_mm = config.CAMERA_FOV_WIDTH_MM
    if fov_h_mm is None:
        fov_h_mm = config.CAMERA_FOV_HEIGHT_MM
    overlap_ratio = _clamp(overlap_pct, OVERLAP_MIN_PCT, OVERLAP_MAX_PCT) / 100.0
    row_mm = fov_h_mm * (1.0 - overlap_ratio)
    col_mm = fov_w_mm * (1.0 - overlap_ratio)
    return row_mm, col_mm, min(row_mm, col_mm)


def surface_key(obj: str, p: dict) -> tuple:
    """prepare_viewpoints 결과를 식별하는 캐시 키.

    WD 가 들어가는 이유: ``camera_positions = positions + normals × WD`` 이고 클러스터링과
    Delaunay 그래프가 전부 그 위에서 돈다. 빠뜨리면 WD 를 바꿔도 캐시 히트로 옛 결과가 나온다.
    row/col 이 따로 들어가는 이유: 순서 때문이 아니다 — lawnmower 도 ``min(row, col)`` 만 써서
    FOV 60×40 과 40×60 은 같은 순서를 낸다. 하지만 캐시된 dict 의 row/col_spacing_m 이 그대로
    h5 ``metadata/row_spacing_mm``/``col_spacing_mm`` 로 저장되므로, 키에서 빼면 60×40 으로 만든
    h5 가 40×60 이라고 기록된다.
    """
    return (
        obj,
        round(p["surface_spacing_mm"], 4),
        round(p["row_spacing_mm"], 4),
        round(p["col_spacing_mm"], 4),
        round(p["working_distance_mm"], 4),
        # '어느 면에 뿌리나' 를 바꾸는 것들 — WD 를 키에 넣은 것과 같은 이유다. 빼면 필터를
        # 바꿔도 캐시 히트로 옛 점이 그대로 나온다.
        p["material_rgb"] or "",
        # 어느 파일에서 온 기하인가 — STEP 은 테셀레이션 밀도까지 결과를 바꾼다.
        p["mesh_file"],
        p["tol_linear"] if p["tol_linear"] is not None else -1.0,
        p["part_name"] or "",
        p["align"],
        # 선택 단계도 결과(점 집합)를 바꾼다 — 키에서 빼면 모드를 바꿔도 캐시 히트다.
        p["selection_mode"],
        round(float(p["selection_target"]), 4),
        # 샘플러가 다르면 점 자체가 다르다.
        p["sampler"],
        round(float(p["fillet_skip_mm"]), 4),
        round(float(p["max_incidence_deg"]), 4),
        round(float(p["dof_mm"]), 4),
        bool(p["filter_bottom"]),
        round(float(p["bottom_angle"]), 4),
        bool(p["filter_interior"]),
        bool(p["filter_occluded"]),
    )


@dataclass(frozen=True)
class ViewpointEntry:
    label: str
    path: Path
    object_name: str
    n: int


def distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """n개의 시각적으로 구분되는 RGB 색을 생성한다.

    황금비 hue 간격으로 인접 rank가 확실히 다른 색이 되게 하고, **색 재사용이 없어**
    성분 수가 많아도 서로 다른 두 성분이 같은 색으로 보이지 않는다.
    """
    out: list[tuple[int, int, int]] = []
    for i in range(max(n, 1)):
        h = (i * 0.618033988749895) % 1.0      # 황금비 → 최대 분리
        s = 0.62 + 0.23 * (i % 3) / 2.0        # 채도 변주
        v = 0.98 - 0.18 * (i % 2)              # 명도 변주
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def discover_objects(data_root: Path) -> list[str]:
    """Object names that have data/{object}/mesh/source.obj."""
    return [p.parent.parent.name for p in sorted(data_root.glob("*/mesh/source.obj"))]


# 읽을 수 있는 메시 형식. .stp/.step 은 cascadio(OpenCASCADE)로 테셀레이션된다.
MESH_SUFFIXES = (".obj", ".stp", ".step", ".ply", ".stl", ".glb")


def discover_meshes(data_root: Path, object_name: str) -> list[Path]:
    """data/{object}/mesh/ 안의 읽을 수 있는 메시 파일들. source.obj 를 맨 앞에 둔다."""
    base = data_root / object_name / "mesh"
    found = [p for p in sorted(base.glob("*"))
             if p.suffix.lower() in MESH_SUFFIXES and not p.name.endswith(".bak")]
    found.sort(key=lambda p: (p.name != "source.obj", p.name))
    return found


def discover_viewpoints(data_root: Path, object_name: str) -> list[ViewpointEntry]:
    """Find data/{object}/viewpoint/*/viewpoints*.h5, labelled '{num}/{file}'."""
    entries: list[ViewpointEntry] = []
    base = data_root / object_name / "viewpoint"
    for path in sorted(base.glob("*/viewpoints*.h5")):
        entries.append(_make_entry(path, object_name, label=f"{path.parent.name}/{path.name}"))
    return entries


def _make_entry(path: Path, object_name: str, label: str) -> ViewpointEntry:
    with h5py.File(path, "r") as f:
        n = int(f["viewpoints"]["positions"].shape[0])
    return ViewpointEntry(label=label, path=path.resolve(), object_name=object_name, n=n)


def load_viewpoint_h5(path: Path) -> dict:
    """Adapt the canonical ViewpointData model to the Studio scene dictionary.

    저장된 cluster_id/path_order 는 읽지 않는다 — 옛 파일에는 남아 있지만 이제 아무도
    소비하지 않고(순서는 GLNS 가 정한다), 화면에 띄우면 "이게 실행 순서" 라는 잘못된
    인상을 준다.
    """
    viewpoint = load_viewpoints_hdf5(path)
    adjacency = None
    if viewpoint.adjacency is not None:
        adjacency = {
            "edges": viewpoint.adjacency.edges,
            "method": viewpoint.adjacency.method,
            "stats": viewpoint.adjacency.stats,
        }
    wd_m = viewpoint.working_distance_m
    camera_positions = viewpoint.positions + viewpoint.normals * wd_m
    return _scene_dict(viewpoint.positions, viewpoint.normals, camera_positions,
                       viewpoint.input_mesh, wd_m, adjacency=adjacency,
                       fov_w_mm=viewpoint.fov_width_mm,
                       fov_h_mm=viewpoint.fov_height_mm)


def _scene_dict(positions, normals, camera_positions, input_mesh, wd_m,
                adjacency=None, fov_w_mm=None, fov_h_mm=None) -> dict:
    return {
        "fov_w_mm": fov_w_mm,
        "fov_h_mm": fov_h_mm,
        "positions": positions,
        "normals": normals,
        "camera_positions": camera_positions,
        "n": len(positions),
        "input_mesh": input_mesh,
        "wd_m": wd_m,
        "adjacency": adjacency,
    }


def material_glb(mesh_path: Path, alpha: float = MATERIAL_ALPHA) -> bytes:
    """소스 메시를 **재질 색을 실은** GLB 로. viser 는 add_glb 로 이걸 그대로 받는다.

    ``load_as_trimesh`` 와 달리 concatenate 하지 않는다 — OBJ+MTL 은 재질별로 geometry 가
    나뉘어 로드되고(sample = 회색 + 초록 둘), 합치는 순간 그 구분이 사라진다. 검사 대상
    재질이 어느 면인지가 곧 **Material RGB 필터가 무엇을 고르는지**라 화면에 남겨야 한다.

    투명도는 viser 파라미터가 아니라 재질에 굽는다 — add_glb 에는 opacity 인자가 없고,
    불투명한 껍데기는 표면점과 간선을 통째로 가린다. glTF 의 alphaMode=BLEND 가 그 자리다.
    같은 이유로 doubleSided 도 여기서 켠다(add_mesh_simple 의 side="double" 대응).

    ※ 텍스처(PNG)는 이 리포의 OBJ/MTL 에 붙어 있지 않고 USD 쪽에만 있다 — 여기서 나오는
      것은 재질의 **색**이지 무늬가 아니다.
    """
    loaded = trimesh.load(mesh_path)
    geometries = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) \
        else [loaded]
    for geom in geometries:
        visual = geom.visual
        color = None
        for source in (getattr(visual, "material", None), visual):
            color = getattr(source, "main_color", None)
            if color is not None:
                break
        rgba = trimesh.visual.color.to_rgba(color if color is not None else MESH_RGB)
        geom.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, alpha],
                alphaMode="BLEND", doubleSided=True))
    return loaded.export(file_type="glb")


def load_as_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"No geometry found in {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)!r}")
    return loaded


def path_tree(path: Path) -> str:
    """절대경로를 트리로 편다 — 패널이 좁아서(24em) 한 줄로는 안 들어간다.

    마크다운은 ``/`` 에서 줄을 못 끊는다(코드 스팬이든 아니든 경로엔 공백이 없다).
    그래서 세그먼트마다 줄을 바꾼다: 줄 하나하나가 짧아져 가로 스크롤이 사라지고,
    깊이가 눈에 들어온다. 펜스 코드블록이라 들여쓰기가 그대로 보존된다.
    """
    parts = list(path.parts)
    if not parts:
        return f"```\n{path}\n```"
    lines = []
    head = parts[0]                      # POSIX 는 '/', Windows 는 드라이브
    if len(parts) > 1 and head in ("/", "\\"):
        head = head + parts[1]           # '/' 만 있는 줄은 낭비다 → 첫 세그먼트와 합친다
        parts = parts[2:]
    else:
        parts = parts[1:]
    lines.append(head)
    for depth, seg in enumerate(parts):
        lines.append("  " * depth + "└ " + seg)
    return "```\n" + "\n".join(lines) + "\n```"


def resolve_mesh_path(data: dict, object_name: str) -> Path | None:
    # Prefer the local mesh: stored ``input_mesh`` is often an absolute path from
    # the container the h5 was generated in (e.g. /root/...), unreadable here.
    candidates = []
    try:
        candidates.append(Path(config.get_mesh_path(object_name, mesh_type="source")))
    except Exception:  # noqa: BLE001
        pass
    if data.get("input_mesh"):
        candidates.append(Path(data["input_mesh"]))
    for c in candidates:
        try:
            if c.exists():
                return c
        except OSError:  # e.g. PermissionError on /root/...
            continue
    return None


# ============================================================================
# Studio
# ============================================================================

class Studio:
    """Holds the viser server, GUI, scene state, and generation caches."""

    def __init__(self, server: viser.ViserServer, objects: list[str],
                 data_root: Path, initial_object: str):
        self.server = server
        self.objects = objects
        self.data_root = data_root

        self.layers: dict[str, list] = {
            "mesh": [], "surface": [], "markers": [], "delaunay": [],
        }
        self.data: dict | None = None
        self.scene_full_mesh = None

        # caches (per object / per (object, surface spacing))
        # 재질이 키에 들어간다 — Material RGB 가 이제 입력칸이라 같은 물체에서 바뀔 수 있고,
        # 그 값이 곧 target_mesh 를 정한다.
        self.mesh_cache: dict[tuple, tuple] = {}  # (obj, material) -> (full, target, input_path)
        self.glb_cache: dict[str, bytes] = {}    # obj -> 재질 보기용 GLB (한 번만 굽는다)
        self._meshes: dict[str, Path] = {}       # 파일명 -> 경로 (Mesh source 드롭다운)
        self.surface_cache: dict[tuple, dict] = {}  # (obj, spacing) -> prepare_viewpoints result
        # (obj, spacing, k, distance_factor, max_normal_angle) -> adjacency dict
        self.adjacency_cache: dict[tuple, dict] = {}
        self.last: dict | None = None            # last generated result, for Save
        self.generating = False
        self._existing: dict[str, ViewpointEntry] = {}
        # 드롭다운 선택을 코드가 바꿀 때 on_update(=디스크 로드)를 막는다. Save 직후
        # 방금 쓴 파일을 선택 상태로 만드는데, 그게 로드로 이어지면 self.last 가 지워져
        # (같은 결과를) 다시 저장할 수 없게 된다.
        self._suppress_existing = False

        self._build_gui(initial_object)
        self._refresh_mesh_options()
        self._refresh_existing_options()

    # ---------- GUI construction ----------
    def _build_gui(self, initial_object: str) -> None:
        g = self.server.gui
        # Object 는 폴더에 넣지 않는다 — control_layout="collapsible" 에서 폴더는 전부 접히는데,
        # 이건 설정이 아니라 **내비게이션**이라 접히면 물체를 바꿀 방법이 사라진다.
        self.object_dd = g.add_dropdown("Object", options=self.objects, initial_value=initial_object)

        initial_overlap = default_overlap_pct()

        # "저장본을 불러온다" 와 "새로 만든다" 는 같은 질문(**이 물체의 viewpoint 를 어디서
        # 얻나**)의 두 답이다. 예전에는 패널 반대편에 떨어져 있었다 — 한 지붕 아래 둔다.
        with g.add_folder("Viewpoints"):
            self.existing_dd = g.add_dropdown(
                "Saved viewpoints", options=[NONE_LABEL], initial_value=NONE_LABEL)

            # 카메라의 물리 스펙만 — 이 셋이 h5 metadata/camera_spec 으로 저장되고, 그 h5 를 읽는
            # IK/궤적/GLNS/Isaac 이 config 대신 이 값을 쓴다. h5 를 로드하면 그 파일 값으로
            # 맞춰진다(_adopt_camera_spec).
            with g.add_folder("Camera spec"):
                self.nb_fov_w = g.add_number(
                    "FOV width (mm)", initial_value=float(config.CAMERA_FOV_WIDTH_MM),
                    min=FOV_MIN_MM, max=FOV_MAX_MM, step=1.0)
                self.nb_fov_h = g.add_number(
                    "FOV height (mm)", initial_value=float(config.CAMERA_FOV_HEIGHT_MM),
                    min=FOV_MIN_MM, max=FOV_MAX_MM, step=1.0)
                # 하한이 물리 제약이다 — 이보다 작으면 검사면이 렌즈 배럴 안쪽에 놓인다.
                self.nb_wd = g.add_number(
                    "Working distance (mm)", initial_value=float(config.CAMERA_WORKING_DISTANCE_MM),
                    min=WD_MIN_MM, max=WD_MAX_MM, step=1.0)
                # 검사 품질 한계 두 개. 이 둘 + 면의 곡률이 '유효 FOV' 를 정한다 —
                # 곡률이 급한 면은 프레임 가장자리가 기울고 멀어져 공칭 FOV 를 다 못 쓴다.
                # 곡면 정의가 필요하므로 CAD faces 샘플러에서만 반영된다.
                self.nb_incidence = g.add_number(
                    "Max incidence (°)", initial_value=float(config.CAMERA_MAX_INCIDENCE_DEG),
                    min=0.0, max=89.0, step=5.0,
                    hint="프레임 가장자리에서 표면이 기울어 보여도 되는 한계. "
                         "0 = 제한 없음 (CAD faces 샘플러 전용)")
                self.nb_dof = g.add_number(
                    "Depth of field ± (mm)", initial_value=float(config.CAMERA_DEPTH_OF_FIELD_MM),
                    min=0.0, max=100.0, step=1.0,
                    hint="초점이 맞는 거리 범위. 0 = 제한 없음 (CAD faces 샘플러 전용)")

            # 어떤 파일의 기하 위에 뿌릴지. 기본은 source.obj 지만, CAD 원본(.stp)을 바로
            # 읽어 비교할 수 있다 — STEP 은 B-rep 이라 삼각형이 없어서 테셀레이션을 거치고,
            # CAD 좌표계를 쓰므로 source.obj 자세/원점에 자동 정렬한다(load_meshes).
            with g.add_folder("Mesh"):
                self.dd_mesh = g.add_dropdown(
                    "Mesh source", options=("source.obj",), initial_value="source.obj",
                    hint="data/{object}/mesh/ 안의 메시 파일. .stp 는 CAD 원본")
                self.nb_tol = g.add_number(
                    "STEP tessellation (mm)", initial_value=float(DEFAULT_STEP_TOL_LINEAR),
                    min=0.01, max=5.0, step=0.05,
                    hint="곡면을 몇 mm 오차로 근사할지 — 작을수록 삼각형이 많다 (.stp 에만 적용)")
                # 어셈블리 STEP 은 지그까지 들어 있다(sample_step: SAMPLE + SAMPLE_BRACKET).
                # 검사 대상만 고르는 자리 — OBJ 의 Material RGB 에 대응한다.
                self.dd_part = g.add_dropdown(
                    "Part", options=(PART_ALL,), initial_value=PART_ALL,
                    hint="파일 안의 부품(솔리드). 고르면 그 부품만 샘플링하고 "
                         "자세·원점도 그 부품 기준으로 맞춘다")
                # STEP 은 CAD 좌표계를 그대로 들고 온다. auto 는 source.obj 와 방향별 면적
                # 분포를 맞추는데, 두 파일이 다른 형상이면(어셈블리 vs 부품) 사실상 동점이
                # 되어 물체가 뒤집힌다 — 그때 손으로 못박으라고 둔 노브다.
                self.dd_align = g.add_dropdown(
                    "Align", options=("auto", "z-up", "y-up", "none"), initial_value="auto",
                    hint="CAD 의 어느 축이 '위' 인가. auto 는 source.obj 와 맞춘다"
                         "(모호하면 콘솔에 경고가 뜬다)")

            # 어느 '면' 에 점을 뿌릴지. 셋 다 원래는 config 표
            # (OBJECT_TARGET_MATERIAL / OBJECT_FILTER_INTERIOR)와 CLI 플래그에만 있어서,
            # 화면만 봐서는 왜 개수가 그렇게 나왔는지 알 수 없었다. 기본값은 여전히 그 표에서
            # 오고(물체를 바꾸면 다시 채워진다), 여기서는 이번 실행만 덮어쓴다.
            with g.add_folder("Faces"):
                self.tb_material = g.add_text(
                    "Material RGB", initial_value=config.OBJECT_TARGET_MATERIAL.get(
                        initial_object) or "",
                    hint="예 '0,255,0' — 그 재질 면만 검사한다. 비우면 메시 전체")
                self.cb_filter_bottom = g.add_checkbox(
                    "Filter bottom faces", initial_value=True,
                    hint="아래를 보는 viewpoint 제거 — 로봇이 밑에서 올려다볼 수 없다")
                self.nb_bottom_angle = g.add_number(
                    "Bottom angle (°)", initial_value=80.0, min=1.0, max=179.0, step=1.0,
                    hint="월드 −z 에서 이 각 안쪽을 보는 면을 버린다 (기본 80)")
                self.cb_filter_interior = g.add_checkbox(
                    "Filter interior (hollow)",
                    initial_value=config.OBJECT_FILTER_INTERIOR.get(initial_object) is not None,
                    hint="속 빈 물체의 안쪽 껍데기 제거 — 오목한 바깥 형상엔 부적합")
                # 법선 필터로는 못 잡는 것: 파인 곳·지그 뒤. 카메라 자리에서 광선을 쏴
                # 실제로 보이는지 묻는다. 가림체는 어셈블리 전체(지그 포함)다.
                self.cb_filter_occluded = g.add_checkbox(
                    "Filter occluded (ray)", initial_value=True,
                    hint="카메라 자리에서 안 보이는 점 제거 — 물체 자신이나 지그에 "
                         "가려지는 경우")

            # 아래 넷은 파이프라인의 **단계** 다: 후보를 만들고 → 그중 일부를 고르고 →
            # 덮였는지 세고 → 남은 점 위에 GLNS 순서 제약 그래프를 만든다. 예전에는 넷이
            # "Generate viewpoints" 한 폴더에 섞여 있어, 그래프 노브(Max edge/normal/k)가
            # 점의 개수나 위치를 바꾼다고 읽혔다 — 그 셋은 점이 다 정해진 **뒤** 에만 쓴다.
            with g.add_folder("Candidates"):
                # 어디에 점을 뿌리나: 삼각형 위(FPS) vs CAD 곡면 위(면별 (u,v) 격자).
                # 후자는 STEP 소스에서만 되고, 법선이 해석적이라 테셀레이션 오차가 없다.
                self.dd_sampler = g.add_dropdown(
                    "Sampler", options=(SAMPLER_FPS, SAMPLER_BREP),
                    initial_value=SAMPLER_FPS,
                    hint="CAD faces 는 Mesh source 가 .stp 일 때만 — 면마다 FOV 격자를 깔고 "
                         "트리밍 경계 안쪽만 남긴다")
                # overlap 은 카메라 속성이 아니라 **샘플링 파라미터**라 h5 camera_spec 이
                # 아니라 여기 산다(ViewpointGenParams 도 camera_spec property 밖에 둔다).
                #
                # CAD faces 에서 겹침은 더 이상 커버리지를 만드는 수단이 아니다: 프레임이
                # 사각형이라 겹침 0 으로도 타일링되고, 곡률·DOF 는 유효 FOV 를 줄이는 쪽으로
                # 이미 반영된다(옛날의 50% 겹침이 그 둘을 근사하려던 대용품이었다). 남은
                # 역할은 오차 여유뿐 — 행마다 u 간격을 그 행에서 재기 때문에, 행 중앙에서
                # 멀어지면 u 눈금이 조금 어긋난다(curved_structure 0%→98.6%, 10%→99.8%).
                # Surface FPS 는 프레임 모형 자체가 없어 여전히 50% 가 필요하다.
                self.nb_overlap = g.add_number(
                    "FOV overlap (%)", initial_value=initial_overlap,
                    min=OVERLAP_MIN_PCT, max=OVERLAP_MAX_PCT, step=1,
                    hint="이웃 촬영이 겹치는 비율 — 간격 = FOV × (1-overlap). CAD faces 는 "
                         "오차 여유라 ≈10% 면 되고, Surface FPS 는 50% 가 필요하다")
                self.nb_fillet = g.add_number(
                    "Fillet skip radius (mm)",
                    initial_value=float(brep.DEFAULT_FILLET_MAX_RADIUS_MM),
                    min=0.0, max=50.0, step=1.0,
                    hint="이보다 반지름이 작은 원통/토러스 면(=모서리 필렛)은 건너뛴다 — "
                         "이웃 면 촬영이 이미 덮는다. 0 이면 모두 샘플링 (CAD faces 전용)")

            # 후보 중 무엇을 쓸지. 격자는 규칙적이라 중복이 남고, greedy 는 커버리지를
            # 지키면서 그 중복을 걷어낸다(curved_structure 52→37점, 커버리지 동일).
            with g.add_folder("Selection"):
                self.dd_selection = g.add_dropdown(
                    "Selection", options=tuple(SELECTION_LABELS), initial_value="전부 사용",
                    hint="Greedy 는 커버리지를 유지하며 최소 집합을 고른다 "
                         "(CAD faces 전용 — 셀이 CAD 면에서 나온다)")
                self.nb_sel_target = g.add_number(
                    "Target coverage (%)", initial_value=100.0, min=50.0, max=100.0, step=1.0,
                    hint="greedy 가 이 커버리지에 도달하면 멈춘다. 후보 전체로도 못 미치면 "
                         "더 보탤 것이 없을 때까지")

            # 생성 결과가 면을 실제로 덮는지 (u,v) 셀로 센다. Selection 과 **같은 판정**
            # (visibility.sees)을 쓰므로 두 숫자가 어긋날 수 없다.
            with g.add_folder("Verify"):
                self.cb_coverage = g.add_checkbox(
                    "Coverage check", initial_value=True,
                    hint="면마다 매개변수 공간을 잘게 나눠 '덮임/덮을 수 없음/구멍' 을 센다 "
                         "(CAD faces 전용, 0.5초 내외)")

            # 점이 다 정해진 뒤, 그 위에 GLNS 의 순서 제약 그래프를 만든다. 점의 개수·위치는
            # 이 셋으로 바뀌지 않는다 — 바뀌는 것은 간선(= GLNS 가 고를 수 있는 이동)뿐.
            #
            # 노브 이름은 알고리즘이 아니라 **무엇의 상한인지**를 말하게 한다. 'delaunay'
            # 접두사는 붙이지 않는다 — 그건 폴더/hint 가 이미 말한다. 앞의 둘이 그래프 '모양'
            # 을 정하고, k 는 '탐색 폭' 이라 성격이 달라 맨 아래에 둔다. 셋 다 슬라이더가
            # 아니라 number 다: 끌어도 Generate 전까지 화면이 바뀌지 않아, 드래그 어포던스가
            # 지키지 못할 약속을 하기 때문이다.
            #
            # 간선 길이 상한을 mm 로 미리 보여주고 싶어지는데, 하지 않는다: factor 는
            # 표면 간격이 아니라 **카메라 위치** 간격에 곱해지고, 카메라는 WD 만큼
            # 떨어져 곡면에서 부챗살처럼 벌어진다(cylinder_sample: 표면 9.5mm vs
            # 카메라 31.1mm, 국소 최대 152mm). 생성 전에는 맞는 값을 낼 수 없다.
            with g.add_folder("Solver graph"):
                self.nb_distfactor = g.add_number(
                    "Max edge length (×)",
                    initial_value=DEFAULT_DELAUNAY_DISTANCE_FACTOR,
                    min=1.0, max=5.0, step=0.1,
                    hint="간선 길이 상한 — 주변 카메라 위치 간격의 배수")
                self.nb_maxangle = g.add_number(
                    "Max normal angle (°)",
                    initial_value=DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
                    min=15, max=180, step=5,
                    hint="두 점의 법선이 이보다 벌어지면 잇지 않는다 (90° = 반대편 면 차단)")
                self.nb_knn = g.add_number(
                    "Neighbor search (k)", initial_value=DEFAULT_DELAUNAY_NEIGHBORS,
                    min=3, max=30, step=1,
                    hint="삼각분할 후보로 볼 이웃 수")

            # 실행과 상태는 네 폴더 **밖** 에 둔다 — 어느 한 단계가 아니라 넷 전부를 돌린다.
            self.btn_generate = g.add_button("Generate")
            self.btn_save = g.add_button("Save h5")
            self.gen_status = g.add_markdown("Idle.")

        # 화면에 무엇을 그릴지 — 순수 토글만 둔다. hops 는 표시가 아니라 데이터를 다시
        # 계산하는 렌즈라 여기가 아니라 진단창 옆에 있다.
        with g.add_folder("Display"):
            self.cb_mesh = g.add_checkbox("Mesh", initial_value=True)
            self.cb_material_view = g.add_checkbox(
                "Material colors", initial_value=False,
                hint="메시를 재질 색으로 그린다 — sample 처럼 대상(초록)/비대상(회색) 재질이 "
                     "나뉜 물체에서 Material RGB 가 무엇을 고르는지 눈으로 확인된다")
            self.cb_surface = g.add_checkbox(
                "Surface points", initial_value=True,
                hint="메시 표면 위의 검사 지점")
            self.cb_markers = g.add_checkbox(
                "Camera positions", initial_value=True,
                hint="표면점 + 법선 × WD — 로봇 EE 가 실제로 가는 곳")
            self.cb_delaunay = g.add_checkbox(
                "Graph edges", initial_value=True,
                hint="GLNS 순서 제약 그래프. 색은 연결 성분")

        # 이 노브가 바꾸는 것(성분 색·GLNS 가 푸는 간선 수)이 바로 아래 진단창과 화면에
        # 있어서 그 옆에 둔다. 슬라이더인 이유: 끌면 즉시 반영된다(Generate 불필요) —
        # Generate 폴더의 숫자칸들과 반대다.
        self.sl_hops = g.add_slider(
            "Solver graph (hops)", min=1, max=MAX_GLNS_HOPS, step=1,
            initial_value=DEFAULT_GLNS_HOPS,
            hint="GLNS 는 저장된 1-hop 간선을 N-hop 으로 확장해 푼다. "
                 "solve.py --delaunay-expand-hops 와 같은 값으로 두세요 (기본 2)")
        self.info = g.add_markdown(IDLE_HINT)

        # callbacks
        self.object_dd.on_update(lambda _: self._on_object_change())
        self.dd_mesh.on_update(lambda _: self._on_mesh_source_change())
        self.existing_dd.on_update(lambda _: self._on_existing_change())
        self.sl_hops.on_update(lambda _: self._on_hops_change())
        for cb in (self.cb_mesh, self.cb_surface, self.cb_markers, self.cb_delaunay):
            cb.on_update(lambda _: self._apply_visibility())
        # 이건 표시/숨김이 아니라 노드 종류(add_glb vs add_mesh_simple)를 바꾼다 → 다시 그린다.
        self.cb_material_view.on_update(lambda _: self._on_material_view_change())
        self.btn_generate.on_click(lambda _: self._on_generate())
        self.btn_save.on_click(lambda _: self._on_save())

    def _expanded_edges(self, adjacency, n) -> tuple[np.ndarray, int]:
        """(hop 확장된 간선, hop 수) — GLNS 가 실제로 푸는 그래프.

        h5 에 저장된 간선은 항상 1-hop 이다. solve.py 가 --delaunay-expand-hops 로 확장한
        뒤에 성분을 세므로, 화면도 같은 것을 보여줘야 "이 물체가 몇 조각인가" 라는 질문에
        같은 답이 나온다.
        """
        edges = np.asarray(adjacency.get("edges", []), dtype=np.int32).reshape(-1, 2)
        hops = int(self.sl_hops.value)
        if hops > 1 and len(edges):
            edges = np.asarray(expand_edges_by_hops(edges, n, hops), dtype=np.int32)
        return edges, hops

    def _current_mesh_path(self) -> Path:
        """Mesh source 드롭다운이 가리키는 파일. 목록이 비면 정규 경로로 떨어진다."""
        chosen = self._meshes.get(self.dd_mesh.value)
        if chosen is not None:
            return chosen
        return Path(config.get_mesh_path(self.object_dd.value, mesh_type="source"))

    def _refresh_mesh_options(self) -> None:
        """물체의 메시 파일 목록을 다시 훑는다. 선택은 source.obj(맨 앞)로 되돌린다."""
        found = discover_meshes(self.data_root, self.object_dd.value)
        self._meshes = {p.name: p for p in found}
        options = list(self._meshes) or ["source.obj"]
        self.dd_mesh.options = options
        self.dd_mesh.value = options[0]
        self._refresh_part_options()

    def _refresh_part_options(self) -> None:
        """지금 소스 파일 안의 부품 목록. 하나뿐이면 고를 것이 없으니 (all) 만 둔다."""
        options = [PART_ALL]
        try:
            # OBJ 는 재질 그룹이 부품처럼 보이지만 그건 Material RGB 가 고르는 것이다 —
            # 같은 것을 두 노브가 고르면 어느 쪽이 이겼는지 화면으로 알 수 없다.
            path = self._current_mesh_path()
            names = list(load_source_parts(path)) \
                if path.suffix.lower() in STEP_SUFFIXES else []
            if len(names) > 1:
                options += names
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] part list failed: {exc}")
        self.dd_part.options = options
        self.dd_part.value = PART_ALL

    def _current_part(self) -> str | None:
        value = self.dd_part.value
        return None if value == PART_ALL else value

    def _on_mesh_source_change(self) -> None:
        """소스 파일이 바뀌면 재질 필터를 그 소스에 맞춘다.

        STEP/CAD 에는 재질 구분이 없다 — 물체 기본값('0,255,0' 같은)이 칸에 남아 있으면
        Generate 가 'No materials matched' 로 실패한다. 파일 종류가 정하는 문제라 사람이
        기억해서 지울 일이 아니다. OBJ 로 돌아오면 물체 기본값을 다시 채운다.
        """
        if self._current_mesh_path().suffix.lower() in STEP_SUFFIXES:
            self.tb_material.value = ""
        else:
            self.tb_material.value = config.OBJECT_TARGET_MATERIAL.get(
                self.object_dd.value) or ""
        self._refresh_part_options()

    def _current_material(self) -> str | None:
        """Material RGB 입력칸 → load_meshes 인자. 빈 칸은 '필터 없음'(None) 이다."""
        return self.tb_material.value.strip() or None

    def _current_overlap_pct(self) -> float:
        return float(self.nb_overlap.value)

    def _current_fov_mm(self) -> tuple[float, float]:
        return float(self.nb_fov_w.value), float(self.nb_fov_h.value)

    def _current_wd_mm(self) -> float:
        return float(self.nb_wd.value)

    def _current_spacing(self) -> tuple[float, float, float]:
        fov_w, fov_h = self._current_fov_mm()
        return fov_spacing_mm(self._current_overlap_pct(), fov_w, fov_h)

    def _refresh_existing_options(self, *, select: str | None = None,
                                  keep_generated: bool = False) -> None:
        """저장본 목록을 다시 훑고, 드롭다운이 **지금 화면의 출처**를 가리키게 한다.

        ``select`` 로 특정 항목(방금 저장한 파일)을, ``keep_generated`` 로 아직 디스크에
        없는 생성 결과를 표시한다. 선택은 콜백을 억제한 채 바꾼다 — 프로그램이 고른 것은
        "불러와라" 가 아니라 "지금 이게 화면에 있다" 는 표시이기 때문이다.
        """
        self._existing = {e.label: e for e in discover_viewpoints(self.data_root, self.object_dd.value)}
        options = [NONE_LABEL] + list(self._existing.keys())
        if keep_generated:
            options.append(GENERATED_LABEL)
        target = select if select in options else NONE_LABEL
        self._suppress_existing = True
        try:
            self.existing_dd.options = options
            self.existing_dd.value = target
        finally:
            self._suppress_existing = False

    def _clear_scene(self) -> None:
        """씬과 거기 딸린 상태를 비운다.

        Object 를 바꿔도 이전 물체의 viewpoint 가 화면에 남아 있었다. 드롭다운이 거짓말을
        하는 것도 문제지만, 그 상태에서 hops 를 건드리면 ``_build_scene`` 이
        **새 물체의 회전**(apply_object_placement)을 **이전 물체의 데이터**에 씌워 물체가
        엉뚱한 자세로 돌아갔다. 비우는 쪽이 정직하고 그 버그도 같이 사라진다.
        """
        self._clear_layers()
        self.data = None
        self.scene_full_mesh = None
        self.last = None          # 화면에 없는 것을 Save 할 수는 없다
        self.info.content = IDLE_HINT

    # ---------- callbacks ----------
    def _on_object_change(self) -> None:
        self._clear_scene()
        self._refresh_existing_options()
        # 낡은 Done/Saved 를 지우는 것이 목적이다 — 안 지우면 상태줄이 이전 물체의 결과를
        # 계속 주장한다(sample 에서 "Done · 74 vp" 를 띄운 채 cylinder 로 갈아타는 식).
        # 안내 문구는 넣지 않는다: info 의 IDLE_HINT 가 같은 말을 하고, 물체 이름은 바로
        # 위 Object 드롭다운이 이미 보여준다.
        self.gen_status.content = "Idle."
        # 면 필터 기본값은 물체별이다 — 이전 물체의 재질/interior 설정을 들고 가면 조용히
        # 틀린 개수가 나온다(sample 의 '0,255,0' 을 들고 cylinder 로 가면 매칭 실패).
        obj = self.object_dd.value
        self._refresh_mesh_options()
        self.tb_material.value = config.OBJECT_TARGET_MATERIAL.get(obj) or ""
        self.cb_filter_interior.value = config.OBJECT_FILTER_INTERIOR.get(obj) is not None

    def _on_existing_change(self) -> None:
        if self._suppress_existing:
            return
        label = self.existing_dd.value
        if label in (NONE_LABEL, GENERATED_LABEL):
            return
        entry = self._existing[label]
        data = load_viewpoint_h5(entry.path)
        mp = resolve_mesh_path(data, entry.object_name)
        full = None
        if mp is not None:
            try:
                full = load_as_trimesh(mp)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] mesh load failed {mp}: {exc}")
        self.last = None  # loaded (not generated) → nothing to Save
        self._adopt_camera_spec(data)
        self._set_scene(full, data, source=f"h5: {label}")

    def _adopt_camera_spec(self, data: dict) -> None:
        """로드한 h5 의 카메라 스펙을 입력칸에 반영한다.

        isaac_pipeline 의 ``_sync_camera_spec_from_h5`` 와 같은 동작 — "기존 것 불러와
        살짝 바꿔 재생성" 이 config 기본값이 아니라 그 파일의 스펙에서 출발하게 한다.
        FOV/WD 입력칸에는 on_update 가 없다 — 대입은 값만 바꾸고, 그 값은 다음 Generate
        때 읽힌다(즉 로드만으로 화면이 다시 그려지지는 않는다).

        overlap 은 여기서 건드리지 않는다 — h5 camera_spec 에 없는 샘플링 파라미터다.
        """
        wd_mm = float(data.get("wd_m") or 0.0) * 1000.0
        if wd_mm > 0.0:
            self.nb_wd.value = _clamp(wd_mm, WD_MIN_MM, WD_MAX_MM)
        for handle, key in ((self.nb_fov_w, "fov_w_mm"), (self.nb_fov_h, "fov_h_mm")):
            value = data.get(key)
            if value:
                handle.value = _clamp(float(value), FOV_MIN_MM, FOV_MAX_MM)

    def _on_generate(self) -> None:
        if self.generating:
            return
        # 입력칸 하한만 믿지 않는다 — add_number 는 타이핑 입력도 받는다.
        problem = config.working_distance_error(self._current_wd_mm())
        if problem:
            self.gen_status.content = f"**Error:** {problem}"
            return
        self.generating = True
        try:
            self.btn_generate.disabled = True
        except Exception:  # noqa: BLE001
            pass
        self.gen_status.content = "⏳ Generating…"
        row_spacing_mm, col_spacing_mm, surface_spacing_mm = self._current_spacing()
        fov_w_mm, fov_h_mm = self._current_fov_mm()
        p = {
            "obj": self.object_dd.value,
            "material_rgb": self._current_material(),
            "mesh_path": self._current_mesh_path(),
            "mesh_file": self._current_mesh_path().name,
            "tol_linear": float(self.nb_tol.value),
            "part_name": self._current_part(),
            "align": self.dd_align.value,
            "sampler": self.dd_sampler.value,
            "fillet_skip_mm": float(self.nb_fillet.value),
            "coverage": bool(self.cb_coverage.value),
            "selection_mode": SELECTION_LABELS.get(self.dd_selection.value,
                                                   select.SELECTION_ALL),
            "selection_target": float(self.nb_sel_target.value) / 100.0,
            "max_incidence_deg": float(self.nb_incidence.value),
            "dof_mm": float(self.nb_dof.value),
            "filter_bottom": bool(self.cb_filter_bottom.value),
            "bottom_angle": float(self.nb_bottom_angle.value),
            "filter_interior": bool(self.cb_filter_interior.value),
            "filter_occluded": bool(self.cb_filter_occluded.value),
            "surface_overlap_pct": self._current_overlap_pct(),
            "surface_spacing_mm": surface_spacing_mm,
            "row_spacing_mm": row_spacing_mm,
            "col_spacing_mm": col_spacing_mm,
            "fov_width_mm": fov_w_mm,
            "fov_height_mm": fov_h_mm,
            "working_distance_mm": self._current_wd_mm(),
            "k_neighbors": int(self.nb_knn.value),
            "distance_factor": float(self.nb_distfactor.value),
            "max_normal_angle_deg": float(self.nb_maxangle.value),
        }
        threading.Thread(target=self._generate_worker, args=(p,), daemon=True).start()

    def _generate_worker(self, p: dict) -> None:
        try:
            obj = p["obj"]
            # 물체 배치를 먼저 반영한다 — bottom filter 가 config.TARGET_OBJECT['rotation'] 으로
            # '아래' 를 판정하기 때문이다. 예전에는 _build_scene 에서만 불러서, 첫 Generate 는
            # 직전에 그려진 물체의 회전으로 필터가 돌았다(지금 배치가 전부 z-yaw 라 결과로는
            # 드러나지 않았을 뿐이다).
            config.apply_object_placement(obj)
            mesh_key = (obj, p["material_rgb"] or "", p["mesh_file"], p["tol_linear"],
                        p["part_name"] or "", p["align"])
            if mesh_key not in self.mesh_cache:
                self.mesh_cache[mesh_key] = load_meshes(
                    obj, p["material_rgb"], mesh_path=p["mesh_path"],
                    part_name=p["part_name"], align=p["align"],
                    tol_linear=p["tol_linear"])
            full_mesh, target_mesh, input_path = self.mesh_cache[mesh_key]

            gkey = surface_key(obj, p)
            if gkey not in self.surface_cache:
                fi = config.OBJECT_FILTER_INTERIOR.get(obj)  # hull_align_min 기본값의 출처
                params = ViewpointGenParams(
                    surface_spacing_mm=p["surface_spacing_mm"],
                    row_spacing_mm=p["row_spacing_mm"],
                    col_spacing_mm=p["col_spacing_mm"],
                    working_distance_mm=p["working_distance_mm"],
                    fov_width_mm=p["fov_width_mm"],
                    fov_height_mm=p["fov_height_mm"],
                    filter_bottom=p["filter_bottom"],
                    bottom_angle=p["bottom_angle"],
                    filter_interior=p["filter_interior"],
                    interior_hull_align_min=(fi or {}).get("hull_align_min", 0.3),
                    filter_occluded=p["filter_occluded"],
                    max_incidence_deg=p["max_incidence_deg"],
                    depth_of_field_mm=p["dof_mm"],
                )
                if p["sampler"] == SAMPLER_BREP:
                    self.surface_cache[gkey] = self._sample_cad_faces(obj, p, params, full_mesh)
                else:
                    # hull 은 자르기 전 전체 메시에서 — 재질 필터로 잘린 조각의 hull 은
                    # 물체의 hull 이 아니다.
                    self.surface_cache[gkey] = prepare_viewpoints(
                        target_mesh, params, hull_mesh=full_mesh,
                        occluder_mesh=full_mesh)
            surface = self.surface_cache[gkey]

            # adjacency 는 클러스터링보다 먼저 — stage1=delaunay 의 입력이기도 하고,
            # 어느 방법이든 파일에 항상 같은 형태로 들어가는 그래프이기 때문이다.
            # 키가 gkey 를 포함해야 한다 — 그래프는 camera_positions(=WD 의존) 위에서 만든다.
            akey = gkey + (p["k_neighbors"],
                           round(p["distance_factor"], 4), round(p["max_normal_angle_deg"], 4))
            if akey not in self.adjacency_cache:
                self.adjacency_cache[akey] = build_local_delaunay_adjacency(
                    surface["camera_positions"], surface["normals"],
                    k_neighbors=p["k_neighbors"],
                    distance_factor=p["distance_factor"],
                    max_normal_angle_deg=p["max_normal_angle_deg"],
                )
            adjacency = self.adjacency_cache[akey]

            if p["obj"] != self.object_dd.value:
                # 생성 중 Object 가 바뀌었다 — 이걸 그리면 _clear_scene 이 막으려던
                # "이전 물체가 새 물체 자리에 그려지는" 상황이 그대로 재현된다.
                self.gen_status.content = (
                    f"**Discarded** — 생성 중 Object 가 `{p['obj']}` → "
                    f"`{self.object_dd.value}` 로 바뀌었습니다. 다시 Generate 하세요.")
                return

            data = _scene_dict(
                surface["positions"], surface["normals"], surface["camera_positions"],
                str(input_path), p["working_distance_mm"] / 1000.0,
                adjacency=adjacency,
                fov_w_mm=p["fov_width_mm"], fov_h_mm=p["fov_height_mm"],
            )
            self.last = {"obj": obj, "surface": surface, "params": p,
                         "n": data["n"], "input_path": input_path,
                         "adjacency": adjacency}
            # 화면에는 결과만 — 어떤 파라미터로 만들었는지는 바로 위 입력칸들이 이미 보여준다.
            self._set_scene(full_mesh, data, source="gen · surface + delaunay")
            self._refresh_existing_options(select=GENERATED_LABEL, keep_generated=True)
            ds = adjacency["stats"]
            tag = " · CAD faces" if p["sampler"] == SAMPLER_BREP else ""
            candidates = surface.get("candidate_count")
            if candidates and candidates != data["n"]:
                tag += f" · 후보 {candidates}개에서 선택"
            cov = surface.get("coverage")
            if cov:
                # 분모는 '검사 가능한 면적' 이다 — 아래를 향하거나 이상적 카메라로도
                # 가려서 안 보이는 곳은 뺐다. 뺀 면적은 **반드시 같이 보여준다**: 조용히
                # 줄이면 커버리지 100% 가 "다 덮었다" 인지 "볼 수 있는 것만 셌다" 인지
                # 구분되지 않는다.
                tag += (f"\n\n커버리지 **{cov['covered_ratio']*100:.1f}%** "
                        f"({cov['covered_cm2']:.0f}/{cov['target_cm2']:.0f} cm²)")
                if cov.get("unreachable_cm2", 0.0) >= 0.5:
                    tag += f" · 접근 불가 {cov['unreachable_cm2']:.0f}cm² 제외"
                holes = [(i, r) for i, r in cov["faces"].items()
                         if r["ratio"] < 0.9 and r["target_cm2"] >= 1.0]
                if holes:
                    worst = sorted(holes, key=lambda kv: kv[1]["ratio"])[:3]
                    tag += " · 구멍: " + ", ".join(
                        f"면{i}({r['ratio']*100:.0f}%, {r['target_cm2']:.0f}cm²)"
                        for i, r in worst)
            self.gen_status.content = (
                f"**Done** · {data['n']} vp · {ds['num_edges']} edges · "
                f"{ds['num_components']} component(s){tag}")
        except Exception as exc:  # noqa: BLE001
            self.gen_status.content = f"**Error:** {exc}"
            print(f"[generate] error: {exc}")
        finally:
            self.generating = False
            try:
                self.btn_generate.disabled = False
            except Exception:  # noqa: BLE001
                pass

    def _sample_cad_faces(self, obj: str, p: dict, params: ViewpointGenParams,
                          full_mesh) -> dict:
        """CAD(B-rep) 면 위 격자 샘플러. 후처리는 FPS 와 **같은 함수**를 지난다.

        격자 간격에는 ``min(FOV_w, FOV_h)`` 를 쓴다 — 면의 (u,v) 축과 카메라의 가로/세로가
        어떻게 맞물릴지는 면마다 다르고, 작은 쪽을 쓰면 어느 방향이든 겹침이 보장된다
        (표면 FPS 의 간격 규약과 같다).
        """
        mesh_path = p["mesh_path"]
        if mesh_path.suffix.lower() not in STEP_SUFFIXES:
            raise ValueError(
                f"'{SAMPLER_BREP}' 샘플러는 STEP 소스가 필요하다 — Mesh source 에서 "
                f".stp 파일을 고르세요 (지금: {mesh_path.name})")
        canonical = Path(config.get_mesh_path(obj, mesh_type="source"))
        reference = load_as_trimesh(canonical) if canonical.exists() else None
        positions, normals, meta = brep.sample_step_aligned(
            mesh_path, reference,
            fov_w_mm=p["fov_width_mm"], fov_h_mm=p["fov_height_mm"],
            overlap=p["surface_overlap_pct"] / 100.0,
            fillet_max_radius_mm=p["fillet_skip_mm"],
            tol_linear=p["tol_linear"], align=p["align"], part_name=p["part_name"],
            working_distance_mm=p["working_distance_mm"],
            max_incidence_deg=p["max_incidence_deg"], dof_mm=p["dof_mm"])
        if len(positions) == 0:
            raise ValueError("CAD 면에서 점이 하나도 나오지 않았다 — "
                             "Fillet skip radius 를 낮춰보세요")
        surface = finalize_viewpoints(
            positions, normals, params, hull_mesh=full_mesh, occluder_mesh=full_mesh,
            # face_id 는 필터를 함께 통과해야 한다 — 커버리지 계산과 h5 추적성이 쓴다.
            # 프레임 축까지 실어 보낸다 — 판정이 촬영 사각형을 사각형으로 보려면 필요하고,
            # 필터가 점을 지울 때 함께 지워져야 어긋나지 않는다.
            extras={"face_id": meta["face_id"],
                    "effective_fov_mm": meta["effective_fov_mm"],
                    "frame_u": meta["frame_u"], "frame_v": meta["frame_v"],
                    "fov_u_mm": meta["fov_u_mm"], "fov_v_mm": meta["fov_v_mm"]})
        surface["candidate_count"] = int(len(surface["positions"]))

        need_cells = len(surface["positions"]) and (
            p["coverage"] or p["selection_mode"] != select.SELECTION_ALL)
        cells = None
        if need_cells:
            try:
                cells = brep.surface_cells(
                    mesh_path, part_name=p["part_name"], tol_linear=p["tol_linear"],
                    align=p["align"], reference_mesh=reference)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] surface cells failed: {exc}")

        spec = visibility.SensorSpec.from_params(params)
        frames = visibility.ViewFrames.from_extras(surface["extras"])
        target_mask = None
        if cells is not None:
            # 커버리지와 **같은 기준**이라야 greedy 가 아무도 못 보는 셀을 쫓지 않는다.
            target_mask = brep.inspectable(
                cells, p["bottom_angle"] if p["filter_bottom"] else 0.0,
                config.TARGET_OBJECT["rotation"], occluder=full_mesh, spec=spec)

        # 선택은 **adjacency 앞**이다 — 그래프는 최종 집합 위에서 만들어야 한다.
        if cells is not None and p["selection_mode"] != select.SELECTION_ALL:
            chosen = select.select(
                p["selection_mode"], cells, surface["positions"], surface["normals"],
                surface["extras"]["effective_fov_mm"], full_mesh, spec,
                mask=target_mask, target_ratio=p["selection_target"], frames=frames)
            if chosen is not None and len(chosen):
                frames = frames[chosen] if frames is not None else None
                # candidate_count 는 **필터를 통과한** 후보 수 그대로 둔다. 샘플러 원본
                # 개수로 덮으면 필터가 지운 것과 선택이 고른 것이 섞여 읽힌다.
                surface = subset_viewpoints(surface, chosen)
                surface["selection_mode"] = p["selection_mode"]

        if p["coverage"] and cells is not None and len(surface["positions"]):
            surface["coverage"] = brep.coverage_report(
                mesh_path, surface["positions"], surface["normals"],
                surface["extras"]["effective_fov_mm"],
                working_distance_mm=p["working_distance_mm"], occluder_mesh=full_mesh,
                max_incidence_deg=p["max_incidence_deg"], dof_mm=p["dof_mm"],
                bottom_angle_deg=p["bottom_angle"] if p["filter_bottom"] else 0.0,
                rotation=config.TARGET_OBJECT["rotation"],
                part_name=p["part_name"], tol_linear=p["tol_linear"],
                align=p["align"], reference_mesh=reference, cells=cells, frames=frames)
        return surface

    def _on_save(self) -> None:
        if self.last is None:
            self.gen_status.content = "Generate first, then Save."
            return
        L = self.last
        obj, surface, p = L["obj"], L["surface"], L["params"]
        # 정규 이름으로 쓴다 — resolve_viewpoint_path 가 가장 먼저 찾는 이름이라, 같은
        # 폴더에 후보가 여럿일 때 mtime 이 다음 단계 입력을 정하는 함정이 생기지 않는다.
        # 경로는 config.DATA_ROOT 가 아니라 **이 앱의 data_root** 에서 만든다 —
        # config.get_viewpoint_path 를 쓰면 --data-root 를 줘도 진짜 data/ 에 쓰고,
        # 목록(discover_viewpoints)은 --data-root 를 보므로 저장한 파일이 목록에
        # 안 나타난다(저장 후 드롭다운이 (none) 으로 남던 원인).
        out_path = self.data_root / obj / "viewpoint" / str(L["n"]) / "viewpoints.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = str(out_path)
        # config 가 아니라 생성에 실제로 쓴 값에서 — 안 그러면 WD 120 으로 만든 h5 가 250 이라고
        # 주장하고, 그걸 읽는 IK/궤적/GLNS/Isaac 이 전부 250 으로 계획한다.
        camera_spec = {
            "fov_width_mm": p["fov_width_mm"],
            "fov_height_mm": p["fov_height_mm"],
            "working_distance_mm": p["working_distance_mm"],
        }
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "input_mesh": str(L["input_path"]),
            # 어느 샘플러가 만든 점인지 — 개수만으로는 구분이 안 된다.
            "method": "brep_grid" if p["sampler"] == SAMPLER_BREP else "surface",
            "sampling_mode": "brep_grid" if p["sampler"] == SAMPLER_BREP else "surface",
            # 어셈블리에서 어느 부품을 봤는지 — 개수가 달라지는 이유가 된다.
            "part_name": p["part_name"] or "",
            "align_mode": p["align"],
            # 선택 단계 — 같은 설정인데 개수가 다른 이유가 된다. **요청값이 아니라 실제로
            # 적용된 값**을 남긴다(FPS 경로는 셀이 없어 선택을 돌리지 않는다).
            "selection_mode": surface.get("selection_mode", select.SELECTION_ALL),
            "candidate_count": int(surface.get("candidate_count", L["n"])),
            # 유효 FOV 를 좁힌 검사 품질 한계 — 개수가 달라지는 이유가 된다(0 = 미사용).
            "max_incidence_deg": p["max_incidence_deg"],
            "depth_of_field_mm": p["dof_mm"],
            "fillet_skip_mm": p["fillet_skip_mm"] if p["sampler"] == SAMPLER_BREP else 0.0,
            "surface_spacing_mm": p["surface_spacing_mm"],
            "row_spacing_mm": surface["row_spacing_m"] * 1000.0,
            "col_spacing_mm": surface["col_spacing_m"] * 1000.0,
            # 0~1 비율로 — config·ViewpointGenParams·cli.py 가 쓰는 단위와 같게 둔다
            # (%는 GUI 표기일 뿐이다).
            "overlap_ratio": p["surface_overlap_pct"] / 100.0,
            # 방문 순서가 없으므로 '경로 길이' 도 없다. greedy NN 베이스라인만 남긴다.
            "nn_path_length_mm": surface["original_path_length_mm"],
            # delaunay k/distance_factor/max_normal_angle 은 여기 중복하지 않는다 —
            # save_viewpoints_hdf5 가 adjacency 그룹 attrs 로 이미 기록한다(cli.py 와 동형).
        }
        coverage = surface.get("coverage")
        if coverage:
            metadata["coverage_ratio"] = float(coverage["covered_ratio"])
            # ratio 만 남기면 분모가 무엇이었는지 알 수 없다 — 같이 적는다.
            metadata["coverage_target_cm2"] = float(coverage["target_cm2"])
            metadata["coverage_unreachable_cm2"] = float(coverage.get("unreachable_cm2", 0.0))
        try:
            save_viewpoints_hdf5(
                surface["positions"], surface["normals"], out, metadata, camera_spec,
                adjacency=L["adjacency"],
            )
            self.gen_status.content = "**Saved**\n\n" + path_tree(out_path)
            self._refresh_existing_options(
                select=f"{out_path.parent.name}/{out_path.name}")
            print(f"[save] wrote {out}")
        except OSError as exc:
            self.gen_status.content = (
                f"**Save failed** ({exc.__class__.__name__})\n\n"
                + path_tree(out_path)
                + "\n\n디렉토리 권한 확인 (root 소유일 수 있음).")
            print(f"[save] {out}: {exc}")

    def _on_hops_change(self) -> None:
        """hop 확장은 재생성 없이 색과 진단만 바꾼다(GLNS 가 볼 그래프로 관점 전환)."""
        if self.data is not None:
            self._build_scene(self.scene_full_mesh, self.data)
            self._refresh_info()

    # ---------- scene ----------
    def _clear_layers(self) -> None:
        for handles in self.layers.values():
            while handles:
                handles.pop().remove()

    def _apply_visibility(self) -> None:
        toggles = {
            "mesh": self.cb_mesh, "surface": self.cb_surface,
            "markers": self.cb_markers, "delaunay": self.cb_delaunay,
        }
        for key, cb in toggles.items():
            for handle in self.layers[key]:
                handle.visible = cb.value

    def _build_scene(self, full_mesh, data: dict) -> None:
        """메시 + 표면점 + 카메라 마커 + Delaunay 그래프. 색은 연결성분이 정한다.

        성분 라벨은 저장하지 않고 간선에서 그때그때 파생한다 — hop 확장을 반영해야
        "GLNS 가 보는 그래프" 와 화면이 같은 답을 낸다.
        """
        self._clear_layers()
        srv = self.server
        # 물체별 config rotation 을 부모 frame(/scene)에 적용 → 물체+viewpoint 가 Isaac 과
        # 동일한 외형으로 회전한다(자식 노드는 object-local 좌표 그대로, frame 이 회전을 입힌다).
        config.apply_object_placement(self.object_dd.value)
        obj_wxyz = np.asarray(config.TARGET_OBJECT["rotation"], dtype=np.float64)
        srv.scene.add_frame("/scene", show_axes=False, wxyz=obj_wxyz, position=(0.0, 0.0, 0.0))
        surf = data["positions"]
        cam = data["camera_positions"]
        n = data["n"]
        adjacency = data.get("adjacency")

        mesh_handle = self._add_mesh_node(full_mesh, data)
        if mesh_handle is not None:
            self.layers["mesh"].append(mesh_handle)
        elif full_mesh is None:
            print("  [warn] no mesh to display; skipping mesh layer")

        if adjacency is not None:
            expanded, _ = self._expanded_edges(adjacency, n)
            _, group_id = components_from_edges(expanded, n)
        else:
            group_id = np.zeros(n, dtype=np.int32)
        group_order = np.unique(group_id)

        palette = distinct_colors(len(group_order))  # 성분별 고유 색 (재사용 없음)
        group_colors = {int(g): palette[rank] for rank, g in enumerate(group_order)}
        for group in group_order:
            idx = np.where(group_id == group)[0]
            if idx.size == 0:
                continue
            rgb = np.array(group_colors[int(group)], dtype=np.uint8)
            self.layers["surface"].append(srv.scene.add_point_cloud(
                f"/scene/surface/g{group}", points=surf[idx],
                colors=np.tile(rgb, (len(idx), 1)),
                point_size=0.0025, point_shape="circle"))
            self.layers["markers"].append(srv.scene.add_point_cloud(
                f"/scene/markers/g{group}", points=cam[idx],
                colors=np.tile(rgb, (len(idx), 1)),
                point_size=0.004, point_shape="circle"))

        if adjacency is not None:
            edges = np.asarray(adjacency.get("edges", []), dtype=np.int32).reshape(-1, 2)
            # 간선 색은 항상 그 성분의 색이다. "성분을 가로지르는 간선" 은 존재할 수 없다 —
            # 성분은 이 간선들(을 확장한 것)에서 파생하고, expand_edges_by_hops 가 원본의
            # 상위집합이라 1-hop 간선의 두 끝점은 언제나 같은 확장 성분에 있다.
            # viser 0.2.11에는 batched line-segment primitive가 없어 edge별 2-point spline을 쓴다.
            for edge_idx, (a, b) in enumerate(edges):
                self.layers["delaunay"].append(srv.scene.add_spline_catmull_rom(
                    f"/scene/delaunay/e{edge_idx}", positions=np.stack([cam[a], cam[b]]),
                    color=group_colors[int(group_id[a])], line_width=1.0))

        self._apply_visibility()

    def _add_mesh_node(self, full_mesh, data: dict):
        """메시 레이어 한 노드. 재질 보기면 GLB, 아니면 지금까지의 반투명 단색이다.

        GLB 는 파일에서 다시 굽는다(캐시) — 화면에 있는 ``full_mesh`` 는 load_meshes 가
        concatenate 한 것이라 재질 구분이 이미 사라진 상태다.
        """
        # STEP/CAD 에는 재질이 없다 — 그 경우 단색으로 떨어진다(파일에서 다시 구우면 정렬
        # 전 CAD 좌표로 돌아가 물체가 엉뚱한 자세로 그려지기도 한다).
        if self.cb_material_view.value and self._current_mesh_path().suffix.lower() == ".obj":
            obj = self.object_dd.value
            try:
                if obj not in self.glb_cache:
                    path = resolve_mesh_path(data, obj)
                    if path is None:
                        raise FileNotFoundError(f"source mesh not found for '{obj}'")
                    self.glb_cache[obj] = material_glb(path)
                return self.server.scene.add_glb("/scene/mesh", glb_data=self.glb_cache[obj])
            except Exception as exc:  # noqa: BLE001
                # 재질 보기가 실패해도 화면이 비면 안 된다 — 단색으로 떨어진다.
                print(f"  [warn] material view failed for {self.object_dd.value}: {exc}")
        if full_mesh is None:
            return None
        return self.server.scene.add_mesh_simple(
            "/scene/mesh",
            vertices=np.asarray(full_mesh.vertices), faces=np.asarray(full_mesh.faces),
            color=MESH_RGB, opacity=0.25, side="double")

    def _on_material_view_change(self) -> None:
        if self.data is not None:
            self._build_scene(self.scene_full_mesh, self.data)

    def _set_scene(self, full_mesh, data: dict, source: str) -> None:
        self.data = data
        self.scene_full_mesh = full_mesh
        self._build_scene(full_mesh, data)
        self._refresh_info()
        print(f"Scene: {source} ({data['n']} vp)")

    def _refresh_info(self) -> None:
        """그래프 한 줄 + 조각났을 때의 경고. 그게 전부다.

        예전에는 Source/Viewpoints/Camera 도 찍었는데 전부 다른 위젯과 중복이었다 —
        출처는 Saved viewpoints 드롭다운이, 카메라는 Camera spec 입력칸이, 개수는
        gen_status 가 이미 말한다.

        경고는 둘 다 뺐다.

        **Fragile(절단점)**: 저장된 h5 18개를 재보니 2-hop 에서는 전부 0개이고
        (2-hop 이 이웃의 이웃을 이어 절단점을 없앤다) 모든 진입점이 2-hop 이라 뜨지 않는
        경고였다. cut_vertices 자체는 core 에 남아 있다 — hops=1 분석용.

        **Split graph**: 처방이 틀렸다. "Max normal angle 을 올려라" 였는데 기본값 90° 는
        임의의 값이 아니라 물리적 경계다(dot(n_i,n_j) >= cos 90° = 0, 같은 반구를 볼 때만
        잇는다). 더 올리면 물체를 관통하는 간선을 만든다 — 필터가 막으려던 바로 그것이다.
        게다가 조각난 것 자체가 대개 문제가 아니다: 18개 중 11개가 2성분 이상이고
        cylinder_sample/132(2성분)는 커버리지 100%, transit 이 511초 중 9.7초였다.
        성분 수는 위 한 줄에 이미 있으니 겁만 주는 줄이었다.
        """
        data = self.data
        if data is None:
            return
        adjacency = data.get("adjacency")
        if adjacency is None:
            self.info.content = (
                "⚠ **No graph** — 이 파일에는 Delaunay 간선이 없어 GLNS 가 거부한다. 재생성 필요.")
            return

        edges = np.asarray(adjacency.get("edges", []), dtype=np.int32).reshape(-1, 2)
        n_components, _ = components_from_edges(edges, data["n"])
        expanded, hops = self._expanded_edges(adjacency, data["n"])
        isolated = int(adjacency.get("stats", {}).get("num_isolated", 0))
        self.info.content = (
            f"`{len(edges)} edges` · `{n_components} component"
            f"{'s' if n_components != 1 else ''}` · `{isolated} isolated` · "
            f"GLNS: `{len(expanded)}` ({hops}-hop)"
        )

    # ---------- external entry ----------
    def load_h5_path(self, path: Path) -> None:
        path = path.resolve()
        object_name = path.parents[2].name if len(path.parents) >= 3 else self.object_dd.value
        data = load_viewpoint_h5(path)
        mp = resolve_mesh_path(data, object_name)
        full = load_as_trimesh(mp) if mp is not None else None
        self.last = None
        self._adopt_camera_spec(data)
        self._set_scene(full, data, source=f"h5: {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive viser studio: generate/visualize viewpoints + Delaunay graph.",
    )
    parser.add_argument("--object", type=str, default=None,
                        help="Initial object to select (default: first discovered).")
    scene_config.add_cli_argument(parser)
    parser.add_argument("--viewpoints", type=Path, default=None,
                        help="Load this viewpoints*.h5 on startup.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 씬을 먼저 — 물체 배치(rotation)가 bottom-filter 판정에 쓰인다.
    scene_config.apply_cli(args, config)
    data_root = args.data_root.resolve()
    objects = discover_objects(data_root)
    if not objects:
        raise SystemExit(f"No objects with mesh/source.obj under {data_root}")
    initial = args.object if args.object in objects else objects[0]

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(
        control_layout="collapsible", control_width="large", dark_mode=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=1.0, height=1.0, plane="xy",
                          cell_size=0.05, section_size=0.25)

    studio = Studio(server, objects, data_root, initial)
    if args.viewpoints is not None:
        if args.viewpoints.exists():
            studio.load_h5_path(args.viewpoints)
        else:
            print(f"[warn] --viewpoints not found: {args.viewpoints}")

    print(f"Objects: {', '.join(objects)}")
    print(f"Open: http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")

    # 예전에는 playback 슬라이더를 굴리려고 여기서 tick 을 돌렸다. 방문 순서가
    # 사라지면서 재생할 것이 없어졌고, 모든 갱신은 GUI 콜백에서 일어난다.
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

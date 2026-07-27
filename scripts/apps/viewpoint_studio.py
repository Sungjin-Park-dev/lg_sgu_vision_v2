#!/usr/bin/env python3
"""Interactive viewpoint studio with viser.

Two ways to put viewpoints on screen, both object-centric:

  * **Generate** — pick an object, tune clustering parameters, and
    regenerate viewpoints in-process via the ``viewpoint/cli.py`` seam
    (``load_meshes`` / ``prepare_grid`` / ``cluster_coacd`` / ``cluster_and_order``).
    Viewpoints are generated with surface sampling only. Surface spacing is derived
    from camera FOV and overlap; CoACD is cached per (object, spacing, threshold)
    so tuning sub-cluster parameters is fast (~2s).
  * **Existing h5** — load a previously saved ``viewpoints*.h5`` for the object.

Clustering is two-stage. **Stage 1** splits the viewpoints into coarse groups — either
Delaunay surface components (default; connectivity-based, deterministic, no convex
decomposition) or CoACD convex parts — and **stage 2** sub-clusters inside each group
with agglomerative/dbscan. Either way the saved h5 has the same shape: geometry, the
edges-only Delaunay graph, and one grouping in ``cluster_id``/``cluster_order``/
``path_order``, with the producer named in ``metadata/clustering_method``.

Rendered elements (same as ``core/viewpoint/visualization.py``):
translucent mesh, per-cluster markers, intra-cluster path lines, inter-cluster
transitions, and — for generated results — translucent CoACD part overlays.
Layers toggle independently; a playback slider scrubs/auto-plays the visit order.
**Color by** switches between the file's own clusters and the raw graph components.

Scope: sampling is fixed to ``surface`` and ordering to ``lawnmower`` in this app.
Grid sampling remains available in ``viewpoint/cli.py`` for CLI/batch use.
Material filtering and bottom-filter tuning are not exposed. Found parameters can
be persisted with **Save** for the downstream plan_trajectory step.

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
from common import config
from core.viewpoint import (
    DEFAULT_DELAUNAY_DISTANCE_FACTOR,
    DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
    DEFAULT_DELAUNAY_NEIGHBORS,
    ViewpointGenParams,
    build_local_delaunay_adjacency,
    cluster_and_order,
    cluster_coacd,
    components_from_edges,
    load_meshes,
    load_viewpoints_hdf5,
    prepare_grid as prepare_viewpoints,
    save_viewpoints_hdf5,
)
from core.viewpoint.visualization import _BOLD_COLORS, _PART_COLORS

HIGHLIGHT_RGB = (255, 235, 59)   # moving playback marker
TRAIL_RGB = (255, 205, 0)        # visited path so far
TRANSITION_RGB = (150, 150, 150)  # inter-cluster lines
DELAUNAY_RGB = (0, 180, 220)
MESH_RGB = (180, 180, 180)
SURFACE_RGB = (255, 255, 255)

EPS_SPACING_FACTOR = 1.5  # dbscan 기본 eps = factor × FOV-derived spacing(mm)
DBSCAN_MIN_SAMPLES = 2
DBSCAN_NORMAL_WEIGHT = 0.0
OVERLAP_MIN_PCT = 20
OVERLAP_MAX_PCT = 90
FOV_MIN_MM = 5.0
FOV_MAX_MM = 500.0
WD_MAX_MM = 800.0
# dbscan eps 상한. FOV 가 커지면 유도 eps 가 여기서 포화해 "eps 가 spacing 을 따라간다"는
# 약속이 조용히 깨지므로, FOV 상한에 맞춰 넉넉히 잡는다.
EPS_MAX_MM = 300
SUBCLUSTER_METHODS = ["agglomerative", "dbscan"]
DEFAULT_SUBCLUSTER_METHOD = "agglomerative"

# Stage 1 — 뷰포인트를 큰 덩어리로 가르는 방법. Stage 2(sub-cluster)는 두 경우 공통.
STAGE1_DELAUNAY = "Delaunay"
STAGE1_COACD = "CoACD"
STAGE1_OPTIONS = [STAGE1_DELAUNAY, STAGE1_COACD]
DEFAULT_STAGE1 = STAGE1_DELAUNAY
STAGE1_KEY = {STAGE1_DELAUNAY: "delaunay", STAGE1_COACD: "coacd"}

# 화면 색을 무엇으로 칠할지 — 저장된 클러스터냐, 그래프의 원시 연결성분이냐.
COLOR_BY_CLUSTERS = "Clusters"
COLOR_BY_COMPONENTS = "Delaunay components"
COLOR_BY_OPTIONS = [COLOR_BY_CLUSTERS, COLOR_BY_COMPONENTS]

# 오브젝트별 기본 타깃 머티리얼 RGB ("R,G,B"). 지정 시 그 재질 면만 샘플링한다.
# (CLI의 --material-rgb 와 동일 경로. 미지정 오브젝트는 전체 메시.)
OBJECT_TARGET_MATERIAL = {
    # 컨벤션: 초록(0,255,0) = 검사대상. 회색(170,163,158)은 비대상이라 제외.
    # (source.obj usemtl 스왑으로 대상 평면을 초록으로 통일. CLI --material-rgb "0,255,0" 와 동일.)
    "sample": "0,255,0",
}


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


def eps_default_mm(surface_spacing_mm: float) -> int:
    eps = EPS_SPACING_FACTOR * surface_spacing_mm
    eps = _clamp(eps, 5.0, EPS_MAX_MM)
    return int(eps + 0.5)


def surface_key(obj: str, p: dict) -> tuple:
    """prepare_grid 결과를 식별하는 캐시 키.

    WD 가 들어가는 이유: ``camera_positions = positions + normals × WD`` 이고 클러스터링과
    Delaunay 그래프가 전부 그 위에서 돈다. 빠뜨리면 WD 를 바꿔도 캐시 히트로 옛 결과가 나온다.
    row/col 이 따로 들어가는 이유: surface spacing 은 ``min(row, col)`` 이라 FOV 60×40 과
    40×60 이 같은 키가 되는데, 캐시된 dict 의 row/col_spacing_m 은 lawnmower 순서를 좌우한다.
    """
    return (
        obj,
        round(p["surface_spacing_mm"], 4),
        round(p["row_spacing_mm"], 4),
        round(p["col_spacing_mm"], 4),
        round(p["working_distance_mm"], 4),
    )


@dataclass(frozen=True)
class ViewpointEntry:
    label: str
    path: Path
    object_name: str
    n: int


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def cluster_rgb(rank: int) -> tuple[int, int, int]:
    return hex_to_rgb(_BOLD_COLORS[rank % len(_BOLD_COLORS)])


def distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """n개의 시각적으로 구분되는 RGB 색을 생성한다.

    황금비 hue 간격으로 인접 rank가 확실히 다른 색이 되게 하고, **색 재사용이 없어**
    클러스터 수가 팔레트(25)를 넘어도 서로 다른 두 클러스터가 같은 색으로 안 보인다.
    (기존 `cluster_rgb`는 25색 순환이라 K>25면 멀리 떨어진 두 클러스터가 같은 색이 됨.)
    """
    out: list[tuple[int, int, int]] = []
    for i in range(max(n, 1)):
        h = (i * 0.618033988749895) % 1.0      # 황금비 → 최대 분리
        s = 0.62 + 0.23 * (i % 3) / 2.0        # 채도 변주
        v = 0.98 - 0.18 * (i % 2)              # 명도 변주
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def part_rgb(j: int) -> tuple[int, int, int]:
    return hex_to_rgb(_PART_COLORS[j % len(_PART_COLORS)])


def discover_objects(data_root: Path) -> list[str]:
    """Object names that have data/{object}/mesh/source.obj."""
    return [p.parent.parent.name for p in sorted(data_root.glob("*/mesh/source.obj"))]


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
    """Adapt the canonical ViewpointData model to the Studio scene dictionary."""
    viewpoint = load_viewpoints_hdf5(path)
    positions = viewpoint.positions
    normals = viewpoint.normals
    n = viewpoint.count
    cluster_id = (
        viewpoint.cluster_id
        if viewpoint.cluster_id is not None
        else np.zeros(n, dtype=np.int32)
    )
    path_order = (
        viewpoint.path_order
        if viewpoint.path_order is not None
        else np.arange(n, dtype=np.int32)
    )
    cluster_order = (
        viewpoint.cluster_order
        if viewpoint.cluster_order is not None
        else np.unique(cluster_id)
    )
    adjacency = None
    if viewpoint.adjacency is not None:
        adjacency = {
            "edges": viewpoint.adjacency.edges,
            "method": viewpoint.adjacency.method,
            "stats": viewpoint.adjacency.stats,
        }
    wd_m = viewpoint.working_distance_m
    input_mesh = viewpoint.input_mesh

    camera_positions = positions + normals * wd_m
    return _scene_dict(positions, normals, camera_positions, cluster_id, cluster_order,
                       path_order, input_mesh, wd_m, adjacency=adjacency,
                       fov_w_mm=viewpoint.fov_width_m * 1000.0,
                       fov_h_mm=viewpoint.fov_height_m * 1000.0)


def _scene_dict(positions, normals, camera_positions, cluster_id, cluster_order,
                path_order, input_mesh, wd_m, adjacency=None,
                fov_w_mm=None, fov_h_mm=None) -> dict:
    return {
        "fov_w_mm": fov_w_mm,
        "fov_h_mm": fov_h_mm,
        "positions": positions,
        "normals": normals,
        "camera_positions": camera_positions,
        "cluster_id": cluster_id,
        "cluster_order": cluster_order,
        "path_order": path_order,
        "order": np.argsort(path_order, kind="stable"),  # global visiting order (indices)
        "n": len(positions),
        "input_mesh": input_mesh,
        "wd_m": wd_m,
        "adjacency": adjacency,
    }


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
            "mesh": [], "surface": [], "markers": [], "paths": [], "transitions": [],
            "delaunay": [], "coacd": [],
        }
        self.play: dict[str, object] = {"highlight": None, "visited": None}
        self.data: dict | None = None
        self.scene_full_mesh = None
        self.scene_coacd_parts = None
        self.pb_pos = 0.0
        self.step_slider = None

        # caches (per object / per (object, surface spacing[, threshold]))
        self.mesh_cache: dict[str, tuple] = {}   # obj -> (full_mesh, target_mesh, input_path)
        self.surface_cache: dict[tuple, dict] = {}  # (obj, spacing) -> prepare_viewpoints result
        self.coacd_cache: dict[tuple, tuple] = {}  # (obj, spacing, threshold) -> (ids, parts)
        # (obj, spacing, k, distance_factor, max_normal_angle) -> adjacency dict
        self.adjacency_cache: dict[tuple, dict] = {}
        self.last: dict | None = None            # last generated result, for Save
        self.generating = False
        self._existing: dict[str, ViewpointEntry] = {}

        self._build_gui(initial_object)
        self._refresh_existing_options()

    # ---------- GUI construction ----------
    def _build_gui(self, initial_object: str) -> None:
        g = self.server.gui
        self.object_dd = g.add_dropdown("Object", options=self.objects, initial_value=initial_object)
        self.existing_dd = g.add_dropdown("Existing h5", options=["(none)"], initial_value="(none)")

        with g.add_folder("Layers"):
            self.colorby_dd = g.add_dropdown(
                "Color by", options=COLOR_BY_OPTIONS, initial_value=COLOR_BY_CLUSTERS)
            self.cb_mesh = g.add_checkbox("Mesh", initial_value=True)
            self.cb_surface = g.add_checkbox("Surface points", initial_value=True)
            self.cb_markers = g.add_checkbox("Markers", initial_value=True)
            self.cb_paths = g.add_checkbox("Cluster paths", initial_value=True)
            self.cb_transitions = g.add_checkbox("Transitions", initial_value=True)
            self.cb_delaunay = g.add_checkbox("Delaunay adjacency", initial_value=True)
            self.cb_coacd = g.add_checkbox("CoACD parts", initial_value=False)

        initial_overlap = default_overlap_pct()
        _, _, initial_spacing = fov_spacing_mm(initial_overlap)
        with g.add_folder("Camera spec"):
            # 여기 값이 곧 h5 metadata/camera_spec 으로 저장되고, 그 h5 를 읽는
            # IK/궤적/GLNS/Isaac 이 config 대신 이 값을 쓴다.
            self.nb_fov_w = g.add_number(
                "FOV width (mm)", initial_value=float(config.CAMERA_FOV_WIDTH_MM),
                min=FOV_MIN_MM, max=FOV_MAX_MM, step=1.0)
            self.nb_fov_h = g.add_number(
                "FOV height (mm)", initial_value=float(config.CAMERA_FOV_HEIGHT_MM),
                min=FOV_MIN_MM, max=FOV_MAX_MM, step=1.0)
            # 하한이 물리 제약이다 — 이보다 작으면 검사면이 렌즈 배럴 안쪽에 놓인다.
            self.nb_wd = g.add_number(
                "Working distance (mm)", initial_value=float(config.CAMERA_WORKING_DISTANCE_MM),
                min=float(int(config.CAMERA_MIN_WORKING_DISTANCE_MM) + 1),
                max=WD_MAX_MM, step=1.0)
            self.sl_overlap = g.add_slider(
                "FOV overlap (%)", min=OVERLAP_MIN_PCT, max=OVERLAP_MAX_PCT,
                step=1, initial_value=initial_overlap)
            self.btn_generate = g.add_button("Generate")
            self.btn_save = g.add_button("Save h5")
            self.gen_status = g.add_markdown("Idle.")

        self.generate_folder = g.add_folder("Generate (surface + stage1 + sub-cluster)")
        with self.generate_folder:
            self.stage1_dd = g.add_dropdown(
                "Stage 1", options=STAGE1_OPTIONS, initial_value=DEFAULT_STAGE1)
            # stage1=CoACD 노브
            self.sl_threshold = g.add_slider("coacd_threshold", min=0.05, max=0.5, step=0.05, initial_value=0.25)
            # stage1=Delaunay 노브 — 그래프 자체의 파라미터라 adjacency 결과도 같이 바뀐다.
            self.sl_knn = g.add_slider(
                "delaunay k_neighbors", min=3, max=30, step=1,
                initial_value=DEFAULT_DELAUNAY_NEIGHBORS)
            self.sl_distfactor = g.add_slider(
                "delaunay distance factor", min=1.0, max=5.0, step=0.1,
                initial_value=DEFAULT_DELAUNAY_DISTANCE_FACTOR)
            self.sl_maxangle = g.add_slider(
                "delaunay max normal angle (deg)", min=15, max=180, step=5,
                initial_value=DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG)
            self.submethod_dd = g.add_dropdown(
                "Sub-cluster", options=SUBCLUSTER_METHODS, initial_value=DEFAULT_SUBCLUSTER_METHOD)
            # agglomerative 노브: 클러스터 최대 지름(mm). complete-linkage로 지름 ≤ 값 보장
            # → 멀리 떨어진 viewpoint가 한 클러스터로 묶이는 것 방지.
            self.sl_maxspan = g.add_slider("max span (mm)", min=50, max=500, step=10, initial_value=250)
            # dbscan 노브 (eps는 FOV-derived surface spacing을 자동 추적)
            self.sl_eps = g.add_slider(
                "eps (mm)", min=5, max=EPS_MAX_MM, step=1,
                initial_value=eps_default_mm(initial_spacing))

        self.playback_folder = g.add_folder("Playback")
        with self.playback_folder:
            self.play_cb = g.add_checkbox("Play", initial_value=False)
            self.speed_slider = g.add_slider("Speed (vp/s)", min=1, max=60, step=1, initial_value=10)
        self._make_step_slider(1)

        self.info = g.add_markdown("Pick an object, then **Generate** — or choose an existing h5.")

        # callbacks
        self.object_dd.on_update(lambda _: self._on_object_change())
        self.existing_dd.on_update(lambda _: self._on_existing_change())
        self.colorby_dd.on_update(lambda _: self._on_colorby_change())
        for cb in (self.cb_mesh, self.cb_surface, self.cb_markers,
                   self.cb_paths, self.cb_transitions, self.cb_delaunay, self.cb_coacd):
            cb.on_update(lambda _: self._apply_visibility())
        self.btn_generate.on_click(lambda _: self._on_generate())
        self.btn_save.on_click(lambda _: self._on_save())
        self.submethod_dd.on_update(lambda _: self._apply_subcluster_visibility())
        self.stage1_dd.on_update(lambda _: self._apply_stage1_visibility())
        for handle in (self.sl_overlap, self.nb_fov_w, self.nb_fov_h, self.nb_wd):
            handle.on_update(lambda _: self._on_camera_spec_change())
        self._apply_subcluster_visibility()
        self._apply_stage1_visibility()

    def _current_overlap_pct(self) -> float:
        return float(self.sl_overlap.value)

    def _current_fov_mm(self) -> tuple[float, float]:
        return float(self.nb_fov_w.value), float(self.nb_fov_h.value)

    def _current_wd_mm(self) -> float:
        return float(self.nb_wd.value)

    def _current_spacing(self) -> tuple[float, float, float]:
        fov_w, fov_h = self._current_fov_mm()
        return fov_spacing_mm(self._current_overlap_pct(), fov_w, fov_h)

    def _on_camera_spec_change(self) -> None:
        """FOV·overlap 이 바뀌면 dbscan eps 기본값을 따라 갱신한다.

        유도값(row/col/surface spacing = FOV × (1-overlap))은 화면에 띄우지 않는다 —
        입력칸이 바로 위에 있어 중복이고, 실제 사용된 값은 생성 시 콘솔에 찍힌다
        (``prepare_grid`` 의 "Row/Col spacing", "Working distance").
        """
        _, _, surface_mm = self._current_spacing()
        self.sl_eps.value = eps_default_mm(surface_mm)

    def _apply_subcluster_visibility(self) -> None:
        """Show only controls relevant to the selected sub-clustering method."""
        method = str(self.submethod_dd.value)
        is_agglomerative = method == "agglomerative"
        is_dbscan = method == "dbscan"

        self.sl_maxspan.visible = is_agglomerative
        self.sl_eps.visible = is_dbscan

    def _apply_stage1_visibility(self) -> None:
        """Show only the stage-1 knobs of the selected clustering method."""
        is_coacd = str(self.stage1_dd.value) == STAGE1_COACD
        self.sl_threshold.visible = is_coacd
        for handle in (self.sl_knn, self.sl_distfactor, self.sl_maxangle):
            handle.visible = not is_coacd

    def _stage1_key(self) -> str:
        """'delaunay' | 'coacd' — cluster_and_order 의 method 접두사."""
        return STAGE1_KEY[str(self.stage1_dd.value)]

    def _make_step_slider(self, n: int) -> None:
        if self.step_slider is not None:
            self.step_slider.remove()
        with self.playback_folder:
            self.step_slider = self.server.gui.add_slider(
                "Step", min=0, max=max(int(n) - 1, 1), step=1, initial_value=0)
        self.step_slider.on_update(lambda _: self._on_step())

    def _refresh_existing_options(self) -> None:
        self._existing = {e.label: e for e in discover_viewpoints(self.data_root, self.object_dd.value)}
        self.existing_dd.options = ["(none)"] + list(self._existing.keys())
        self.existing_dd.value = "(none)"

    # ---------- callbacks ----------
    def _on_object_change(self) -> None:
        self._refresh_existing_options()
        self.gen_status.content = f"Object **{self.object_dd.value}** — Generate or pick existing h5."

    def _on_existing_change(self) -> None:
        label = self.existing_dd.value
        if label == "(none)":
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
        self._set_scene(full, data, coacd_parts=None, source=f"h5: {label}")

    def _adopt_camera_spec(self, data: dict) -> None:
        """로드한 h5 의 카메라 스펙을 입력칸에 반영한다.

        isaac_pipeline 의 ``_sync_camera_spec_from_h5`` 와 같은 동작 — "기존 것 불러와
        살짝 바꿔 재생성" 이 config 기본값이 아니라 그 파일의 스펙에서 출발하게 한다.
        ``.value`` 대입이 ``_on_camera_spec_change`` 를 트리거하지만 eps 기본값만 다시 계산한다.
        """
        wd_mm = float(data.get("wd_m") or 0.0) * 1000.0
        if wd_mm > 0.0:
            self.nb_wd.value = _clamp(
                wd_mm, float(int(config.CAMERA_MIN_WORKING_DISTANCE_MM) + 1), WD_MAX_MM)
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
        submethod = str(self.submethod_dd.value)  # 'agglomerative' | 'dbscan'
        if submethod not in SUBCLUSTER_METHODS:
            submethod = DEFAULT_SUBCLUSTER_METHOD
        stage1 = self._stage1_key()               # 'delaunay' | 'coacd'
        row_spacing_mm, col_spacing_mm, surface_spacing_mm = self._current_spacing()
        fov_w_mm, fov_h_mm = self._current_fov_mm()
        p = {
            "obj": self.object_dd.value,
            "sampling_mode": "surface",
            "ordering_mode": "lawnmower",
            "surface_overlap_pct": self._current_overlap_pct(),
            "surface_spacing_mm": surface_spacing_mm,
            "row_spacing_mm": row_spacing_mm,
            "col_spacing_mm": col_spacing_mm,
            "fov_width_mm": fov_w_mm,
            "fov_height_mm": fov_h_mm,
            "working_distance_mm": self._current_wd_mm(),
            "stage1": stage1,
            "submethod": submethod,
            "method": f"{stage1}+{submethod}",
            "threshold": float(self.sl_threshold.value),
            "k_neighbors": int(self.sl_knn.value),
            "distance_factor": float(self.sl_distfactor.value),
            "max_normal_angle_deg": float(self.sl_maxangle.value),
            "max_span_mm": float(self.sl_maxspan.value),
            "eps_mm": float(self.sl_eps.value),
            "normal_weight": DBSCAN_NORMAL_WEIGHT,
            "min_samples": DBSCAN_MIN_SAMPLES,
        }
        threading.Thread(target=self._generate_worker, args=(p,), daemon=True).start()

    def _generate_worker(self, p: dict) -> None:
        try:
            obj = p["obj"]
            if obj not in self.mesh_cache:
                mat = OBJECT_TARGET_MATERIAL.get(obj)  # 예: sample → 초록만. 미지정 시 전체 메시
                self.mesh_cache[obj] = load_meshes(obj, mat)
            full_mesh, target_mesh, input_path = self.mesh_cache[obj]

            sp = p["surface_spacing_mm"]
            gkey = surface_key(obj, p)
            if gkey not in self.surface_cache:
                fi = config.OBJECT_FILTER_INTERIOR.get(obj)  # hollow 물체만 opt-in
                self.surface_cache[gkey] = prepare_viewpoints(
                    target_mesh,
                    ViewpointGenParams(
                        sampling_mode="surface",
                        ordering_mode="lawnmower",
                        surface_spacing_mm=sp,
                        row_spacing_mm=p["row_spacing_mm"],
                        col_spacing_mm=p["col_spacing_mm"],
                        working_distance_mm=p["working_distance_mm"],
                        fov_width_mm=p["fov_width_mm"],
                        fov_height_mm=p["fov_height_mm"],
                        filter_interior=fi is not None,
                        interior_hull_align_min=(fi or {}).get("hull_align_min", 0.3),
                    ),
                )
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

            method = p["method"]  # {delaunay|coacd}+{agglomerative|dbscan}
            common = dict(
                positions=surface["positions"], normals=surface["normals"],
                camera_positions=surface["camera_positions"], target_mesh=target_mesh,
                row_spacing_m=surface["row_spacing_m"], col_spacing_m=surface["col_spacing_m"],
                grid_row_index=surface["grid_row_index"],
                cam_axis1=surface["cam_axis1"], cam_axis2=surface["cam_axis2"],
                original_path_length_mm=surface["original_path_length_mm"],
                normal_weight=p["normal_weight"], ordering_mode=p["ordering_mode"],
                adjacency_edges=adjacency["edges"],
            )
            if p["stage1"] == "coacd":
                # CoACD 만 stage1 이 비싸다 — gkey + threshold 로 캐싱. CoACD 자체는 표면
                # positions 만 쓰므로 엄밀히는 WD 독립이지만, "gkey 중 위치에 영향을 주는
                # 부분집합" 이라는 어디에도 안 적힌 불변식을 만드느니 ~2s 과잉 무효화를 받는다.
                ckey = gkey + (round(p["threshold"], 4),)
                if ckey not in self.coacd_cache:
                    self.coacd_cache[ckey] = cluster_coacd(
                        target_mesh, surface["positions"], p["threshold"])
                common.update(threshold=p["threshold"],
                              precomputed_coacd=self.coacd_cache[ckey])

            if p["submethod"] == "agglomerative":
                result = cluster_and_order(method, method, **common, max_span_mm=p["max_span_mm"])
            elif p["submethod"] == "dbscan":
                result = cluster_and_order(
                    method, method, **common,
                    eps_m=p["eps_mm"] / 1000.0, min_samples=p["min_samples"])
            else:
                raise ValueError(f"Unsupported sub-cluster method in studio: {p['submethod']}")

            data = _scene_dict(
                surface["positions"], surface["normals"], surface["camera_positions"],
                result["cluster_ids"], result["cluster_order"], result["path_order"],
                str(input_path), p["working_distance_mm"] / 1000.0,
                adjacency=adjacency,
                fov_w_mm=p["fov_width_mm"], fov_h_mm=p["fov_height_mm"],
            )
            self.last = {"obj": obj, "surface": surface, "result": result,
                         "params": p, "n": data["n"], "input_path": input_path,
                         "adjacency": adjacency}
            red = (1 - result["path_length_mm"] / surface["original_path_length_mm"]) * 100
            # 화면에는 결과만 — 어떤 파라미터로 만들었는지는 바로 위 입력칸들이 이미 보여준다.
            # 전체 파라미터는 콘솔과 저장된 h5 의 metadata 에 남는다.
            self._set_scene(
                full_mesh, data, coacd_parts=result.get("coacd_parts"),
                source=f"gen · {method}",
            )
            self.gen_status.content = (
                f"**Done** · {data['n']} vp · {result['num_clusters']} clusters · "
                f"path {result['path_length_mm']:.0f} mm ({red:.1f}%)")
        except Exception as exc:  # noqa: BLE001
            self.gen_status.content = f"**Error:** {exc}"
            print(f"[generate] error: {exc}")
        finally:
            self.generating = False
            try:
                self.btn_generate.disabled = False
            except Exception:  # noqa: BLE001
                pass

    def _on_save(self) -> None:
        if self.last is None:
            self.gen_status.content = "Generate first, then Save."
            return
        L = self.last
        obj, surface, result, p = L["obj"], L["surface"], L["result"], L["params"]
        clmethod = p.get("method", "delaunay+agglomerative")
        out = str(config.get_viewpoint_path(obj, L["n"], filename=f"viewpoints_{clmethod}.h5"))
        # config 가 아니라 생성에 실제로 쓴 값에서 — 안 그러면 WD 120 으로 만든 h5 가 250 이라고
        # 주장하고, 그걸 읽는 IK/궤적/GLNS/Isaac 이 전부 250 으로 계획한다.
        camera_spec = {
            "fov_width_mm": p["fov_width_mm"],
            "fov_height_mm": p["fov_height_mm"],
            "working_distance_mm": p["working_distance_mm"],
        }
        sm = p.get("sampling_mode", "surface")
        om = p.get("ordering_mode", "lawnmower")
        sp = p.get("surface_spacing_mm")
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "input_mesh": str(L["input_path"]),
            "method": f"{sm}+{om}",
            "sampling_mode": sm,
            "ordering_mode": om,
            "row_spacing_mm": surface["row_spacing_m"] * 1000.0,
            "col_spacing_mm": surface["col_spacing_m"] * 1000.0,
            "total_path_length_mm": result["path_length_mm"],
        }
        if sp is not None:
            metadata["surface_spacing_mm"] = sp
            metadata["surface_overlap_pct"] = p.get("surface_overlap_pct")
        cluster_meta = {
            "clustering_method": clmethod,
            "num_clusters": result["num_clusters"],
            "clustered_path_length_mm": result["path_length_mm"],
            "original_path_length_mm": surface["original_path_length_mm"],
            "clustering_timestamp": datetime.now().isoformat(),
        }
        # stage-1 파라미터는 실제로 쓴 방법의 것만 기록한다.
        if p.get("stage1") == "coacd":
            cluster_meta["coacd_threshold"] = p["threshold"]
        else:
            cluster_meta["delaunay_k_neighbors"] = p["k_neighbors"]
            cluster_meta["delaunay_distance_factor"] = p["distance_factor"]
            cluster_meta["delaunay_max_normal_angle_deg"] = p["max_normal_angle_deg"]
        if p.get("submethod") == "agglomerative":
            cluster_meta["max_span_mm"] = p["max_span_mm"]
        elif p.get("submethod") == "dbscan":
            cluster_meta["dbscan_eps_mm"] = p["eps_mm"]
            cluster_meta["dbscan_min_samples"] = p["min_samples"]
            cluster_meta["dbscan_normal_weight"] = p["normal_weight"]
        try:
            save_viewpoints_hdf5(
                surface["positions"], surface["normals"], out, metadata, camera_spec,
                result["path_order"],
                cluster_id=result["cluster_ids"], cluster_order=result["cluster_order"],
                cluster_metadata=cluster_meta, adjacency=L["adjacency"],
            )
            self.gen_status.content = f"**Saved** → `{out}`"
            self._refresh_existing_options()
            print(f"[save] wrote {out}")
        except OSError as exc:
            self.gen_status.content = (
                f"**Save failed** ({exc.__class__.__name__}) → `{out}`\n\n"
                f"디렉토리 권한 확인 (root 소유일 수 있음).")
            print(f"[save] {exc}")

    def _on_step(self) -> None:
        self.pb_pos = float(self.step_slider.value)
        self._update_highlight(self.step_slider.value)

    def _on_colorby_change(self) -> None:
        """Recolor the scene for the selected grouping (no regeneration)."""
        if self.data is not None:
            self._build_scene(self.scene_full_mesh, self.data, self.scene_coacd_parts)
            self._refresh_info()
            self._update_highlight(int(self.step_slider.value))

    # ---------- scene ----------
    def _clear_layers(self) -> None:
        for handles in self.layers.values():
            while handles:
                handles.pop().remove()
        for key in ("highlight", "visited"):
            if self.play[key] is not None:
                self.play[key].remove()
            self.play[key] = None

    def _apply_visibility(self) -> None:
        toggles = {
            "mesh": self.cb_mesh, "surface": self.cb_surface,
            "markers": self.cb_markers, "paths": self.cb_paths,
            "transitions": self.cb_transitions, "delaunay": self.cb_delaunay,
            "coacd": self.cb_coacd,
        }
        for key, cb in toggles.items():
            for handle in self.layers[key]:
                handle.visible = cb.value

    def _build_scene(self, full_mesh, data: dict, coacd_parts) -> None:
        self._clear_layers()
        srv = self.server
        # 물체별 config rotation 을 부모 frame(/scene, /play)에 적용 → 물체+viewpoint 가 Isaac 과
        # 동일한 외형으로 회전한다(자식 노드는 object-local 좌표 그대로, frame 이 회전을 입힌다).
        config.apply_object_placement(self.object_dd.value)
        obj_wxyz = np.asarray(config.TARGET_OBJECT["rotation"], dtype=np.float64)
        srv.scene.add_frame("/scene", show_axes=False, wxyz=obj_wxyz, position=(0.0, 0.0, 0.0))
        srv.scene.add_frame("/play", show_axes=False, wxyz=obj_wxyz, position=(0.0, 0.0, 0.0))
        surf = data["positions"]
        cam = data["camera_positions"]
        cid = data["cluster_id"]
        corder = data["cluster_order"]
        porder = data["path_order"]
        adjacency = data.get("adjacency")
        # 성분별 색칠은 저장된 라벨이 아니라 edges 에서 그때그때 파생한다.
        use_components = (self.colorby_dd.value == COLOR_BY_COMPONENTS
                          and adjacency is not None)

        if full_mesh is not None:
            self.layers["mesh"].append(srv.scene.add_mesh_simple(
                "/scene/mesh",
                vertices=np.asarray(full_mesh.vertices), faces=np.asarray(full_mesh.faces),
                color=MESH_RGB, opacity=0.25, side="double"))
        else:
            print("  [warn] no mesh to display; skipping mesh layer")

        if use_components:
            _, group_id = components_from_edges(adjacency["edges"], data["n"])
            group_order = np.unique(group_id)
        else:
            group_id = cid
            group_order = corder

        palette = distinct_colors(len(group_order))  # grouping별 고유 색 (재사용 없음)
        group_colors = {int(group): palette[rank] for rank, group in enumerate(group_order)}
        for group in group_order:
            idx = np.where(group_id == group)[0]
            if idx.size == 0:
                continue
            rgb = group_colors[int(group)]
            self.layers["surface"].append(srv.scene.add_point_cloud(
                f"/scene/surface/g{group}", points=surf[idx],
                colors=np.tile(np.array(rgb, dtype=np.uint8), (len(idx), 1)),
                point_size=0.0025, point_shape="circle"))
            self.layers["markers"].append(srv.scene.add_point_cloud(
                f"/scene/markers/g{group}", points=cam[idx],
                colors=np.tile(np.array(rgb, dtype=np.uint8), (len(idx), 1)),
                point_size=0.004, point_shape="circle"))
            if not use_components:
                ordered = idx[np.argsort(porder[idx], kind="stable")]
                if ordered.size > 1:
                    self.layers["paths"].append(srv.scene.add_spline_catmull_rom(
                        f"/scene/paths/g{group}", positions=cam[ordered],
                        color=rgb, line_width=3.0, curve_type="catmullrom"))

        if not use_components:
            for i in range(len(corder) - 1):
                fi = np.where(cid == corder[i])[0]
                ti = np.where(cid == corder[i + 1])[0]
                if fi.size == 0 or ti.size == 0:
                    continue
                p1 = cam[fi[np.argmax(porder[fi])]]
                p2 = cam[ti[np.argmin(porder[ti])]]
                self.layers["transitions"].append(srv.scene.add_spline_catmull_rom(
                    f"/scene/transitions/t{i}", positions=np.stack([p1, p2]),
                    color=TRANSITION_RGB, line_width=2.0))

        if adjacency is not None:
            edges = np.asarray(adjacency.get("edges", []), dtype=np.int32).reshape(-1, 2)
            # viser 0.2.11에는 batched line-segment primitive가 없어 edge별 2-point spline을 쓴다.
            for edge_idx, (a, b) in enumerate(edges):
                edge_color = (group_colors[int(group_id[a])]
                              if use_components and group_id[a] == group_id[b]
                              else DELAUNAY_RGB)
                self.layers["delaunay"].append(srv.scene.add_spline_catmull_rom(
                    f"/scene/delaunay/e{edge_idx}", positions=np.stack([cam[a], cam[b]]),
                    color=edge_color, line_width=1.0))

        if coacd_parts and not use_components:
            for j, part in enumerate(coacd_parts):
                self.layers["coacd"].append(srv.scene.add_mesh_simple(
                    f"/scene/coacd/p{j}",
                    vertices=np.asarray(part.vertices), faces=np.asarray(part.faces),
                    color=part_rgb(j), opacity=0.3, side="double"))

        self._apply_visibility()

    def _set_scene(self, full_mesh, data: dict, coacd_parts, source: str) -> None:
        self.data = data
        self.scene_full_mesh = full_mesh
        self.scene_coacd_parts = coacd_parts
        self.scene_source = source
        self._build_scene(full_mesh, data, coacd_parts)
        self._make_step_slider(data["n"])
        self.pb_pos = 0.0
        self._update_highlight(0)
        self._refresh_info()
        print(f"Scene: {source} ({data['n']} vp, {len(data['cluster_order'])} clusters)")

    def _refresh_info(self) -> None:
        data = self.data
        if data is None:
            return
        adjacency = data.get("adjacency")
        adjacency_info = ""
        if adjacency is not None:
            edges = np.asarray(adjacency.get("edges", []), dtype=np.int32).reshape(-1, 2)
            n_components, _ = components_from_edges(edges, data["n"])
            adjacency_info = (
                f"**Delaunay:** `{len(edges)} edges` · "
                f"`{n_components} components` · "
                f"`{int(adjacency.get('stats', {}).get('num_isolated', 0))} isolated`"
            )
        lines = [
            f"**Source:** `{self.scene_source}`",
            f"**Color by:** `{self.colorby_dd.value}`",
            f"**Viewpoints:** `{data['n']}`",
            f"**Clusters:** `{len(data['cluster_order'])}`",
            f"**Camera:** `WD {data['wd_m'] * 1000:.0f} mm`"
            + (f" · `FOV {data['fov_w_mm']:.0f}×{data['fov_h_mm']:.0f} mm`"
               if data.get("fov_w_mm") else ""),
        ]
        if adjacency_info:
            lines.append(adjacency_info)
        elif self.colorby_dd.value == COLOR_BY_COMPONENTS:
            lines.append("**Delaunay:** adjacency 그래프가 없어 클러스터 색으로 표시합니다")
        self.info.content = "\n".join(lines)

    def _update_highlight(self, step: int) -> None:
        data = self.data
        if data is None or data["n"] == 0:
            return
        cam = data["camera_positions"]
        order = data["order"]
        step = int(np.clip(step, 0, data["n"] - 1))
        i = int(order[step])

        self.play["highlight"] = self.server.scene.add_point_cloud(
            "/play/highlight", points=cam[i:i + 1],
            colors=np.array([HIGHLIGHT_RGB], dtype=np.uint8),
            point_size=0.012, point_shape="circle")
        if self.play["visited"] is not None:
            self.play["visited"].remove()
            self.play["visited"] = None
        visited = cam[order[:step + 1]]
        if len(visited) >= 2:
            self.play["visited"] = self.server.scene.add_spline_catmull_rom(
                "/play/visited", positions=visited,
                color=TRAIL_RGB, line_width=4.0, curve_type="catmullrom")

    def tick(self, dt: float) -> None:
        data = self.data
        if self.play_cb.value and data is not None and data["n"] > 1:
            self.pb_pos = (self.pb_pos + dt * float(self.speed_slider.value)) % data["n"]
            step = int(self.pb_pos)
            if step != self.step_slider.value:
                self.step_slider.value = step
                self._update_highlight(step)

    # ---------- external entry ----------
    def load_h5_path(self, path: Path) -> None:
        path = path.resolve()
        object_name = path.parents[2].name if len(path.parents) >= 3 else self.object_dd.value
        data = load_viewpoint_h5(path)
        mp = resolve_mesh_path(data, object_name)
        full = load_as_trimesh(mp) if mp is not None else None
        self.last = None
        self._adopt_camera_spec(data)
        self._set_scene(full, data, coacd_parts=None, source=f"h5: {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive viser studio: generate/visualize viewpoints + clusters + path.",
    )
    parser.add_argument("--object", type=str, default=None,
                        help="Initial object to select (default: first discovered).")
    parser.add_argument("--viewpoints", type=Path, default=None,
                        help="Load this viewpoints*.h5 on startup.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    objects = discover_objects(data_root)
    if not objects:
        raise SystemExit(f"No objects with mesh/source.obj under {data_root}")
    initial = args.object if args.object in objects else objects[0]

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(
        control_layout="collapsible", control_width="medium", dark_mode=True)
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

    last_t = time.time()
    try:
        while True:
            now = time.time()
            dt = now - last_t
            last_t = now
            studio.tick(dt)
            time.sleep(0.05)
    except KeyboardInterrupt:
        server.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

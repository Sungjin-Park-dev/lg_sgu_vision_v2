#!/usr/bin/env python3
"""Interactive viewpoint studio with viser.

Two ways to put viewpoints on screen, both object-centric:

  * **Generate** — pick an object, tune sampling and graph parameters, and
    regenerate in-process via the ``viewpoint/cli.py`` seam
    (``load_meshes`` / ``prepare_viewpoints`` / ``build_local_delaunay_adjacency``).
    Surface FPS is the only sampler; spacing comes from camera FOV and overlap.
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

Scope: material filtering and bottom-filter tuning are not exposed. Found
parameters can be persisted with **Save** for the GLNS solve step.

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
    DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
    DEFAULT_DELAUNAY_NEIGHBORS,
    ViewpointGenParams,
    build_local_delaunay_adjacency,
    components_from_edges,
    expand_edges_by_hops,
    load_meshes,
    load_viewpoints_hdf5,
    prepare_viewpoints,
    save_viewpoints_hdf5,
)

MESH_RGB = (180, 180, 180)
SURFACE_RGB = (255, 255, 255)

OVERLAP_MIN_PCT = 20
OVERLAP_MAX_PCT = 90
FOV_MIN_MM = 5.0
FOV_MAX_MM = 500.0
WD_MAX_MM = 800.0
# WD 하한은 물리 제약(검사면이 렌즈 배럴보다 앞에 있어야 한다)에서 온다. config 의 값은
# 경계 자체(그 값이면 렌즈 앞면과 정확히 겹침)라 입력칸 하한은 1mm 안쪽으로 올린다 —
# working_distance_error 가 경계를 실패로 보므로 하한이 곧 유효한 최솟값이 된다.
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
            "mesh": [], "surface": [], "markers": [], "delaunay": [],
        }
        self.data: dict | None = None
        self.scene_full_mesh = None

        # caches (per object / per (object, surface spacing))
        self.mesh_cache: dict[str, tuple] = {}   # obj -> (full_mesh, target_mesh, input_path)
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

            self.generate_folder = g.add_folder("Generate viewpoints")
            with self.generate_folder:
                # 노브 이름은 알고리즘이 아니라 **무엇의 상한인지**를 말하게 한다.
                # 'delaunay' 접두사는 붙이지 않는다 — 그건 폴더/hint 가 이미 말한다.
                # 넷 다 슬라이더가 아니라 number 다: 끌어도 Generate 전까지 화면이
                # 바뀌지 않아, 드래그 어포던스가 지키지 못할 약속을 하기 때문이다.
                #
                # overlap 은 카메라 속성이 아니라 **샘플링 파라미터**라 h5 camera_spec 이
                # 아니라 여기 산다(ViewpointGenParams 도 camera_spec property 밖에 둔다).
                self.nb_overlap = g.add_number(
                    "FOV overlap (%)", initial_value=initial_overlap,
                    min=OVERLAP_MIN_PCT, max=OVERLAP_MAX_PCT, step=1,
                    hint="이웃 촬영 영역이 겹치는 비율 — 표면 점 간격 = min(FOV) × (1-overlap)")
                # 아래 셋이 GLNS 의 순서 제약 그래프를 만든다. 앞의 둘이 그래프 '모양' 을
                # 정하고, k 는 '탐색 폭' 이라 성격이 달라 맨 아래에 둔다.
                #
                # 간선 길이 상한을 mm 로 미리 보여주고 싶어지는데, 하지 않는다: factor 는
                # 표면 간격이 아니라 **카메라 위치** 간격에 곱해지고, 카메라는 WD 만큼
                # 떨어져 곡면에서 부챗살처럼 벌어진다(cylinder_sample: 표면 9.5mm vs
                # 카메라 31.1mm, 국소 최대 152mm). 생성 전에는 맞는 값을 낼 수 없다.
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
                # 실행과 상태는 자기가 쓰는 노브 바로 아래에 둔다.
                self.btn_generate = g.add_button("Generate")
                self.btn_save = g.add_button("Save h5")
                self.gen_status = g.add_markdown("Idle.")

        # 화면에 무엇을 그릴지 — 순수 토글만 둔다. hops 는 표시가 아니라 데이터를 다시
        # 계산하는 렌즈라 여기가 아니라 진단창 옆에 있다.
        with g.add_folder("Display"):
            self.cb_mesh = g.add_checkbox("Mesh", initial_value=True)
            self.cb_surface = g.add_checkbox(
                "Surface points", initial_value=True,
                hint="메시 표면 위의 검사 지점")
            self.cb_markers = g.add_checkbox(
                "Camera positions", initial_value=True,
                hint="표면점 + 법선 × WD — 로봇 EE 가 실제로 가는 곳")
            self.cb_delaunay = g.add_checkbox(
                "Graph edges", initial_value=True,
                hint="GLNS 순서 제약 그래프. 색은 연결 성분")

        # 이 노브가 바꾸는 것(성분 색·간선 수·Fragile 목록)이 전부 바로 아래 진단창에
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
        self.existing_dd.on_update(lambda _: self._on_existing_change())
        self.sl_hops.on_update(lambda _: self._on_hops_change())
        for cb in (self.cb_mesh, self.cb_surface, self.cb_markers, self.cb_delaunay):
            cb.on_update(lambda _: self._apply_visibility())
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
        하는 것도 문제지만, 그 상태에서 Color by 나 hops 를 건드리면 ``_build_scene`` 이
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
        ``.value`` 대입이 ``_on_camera_spec_change`` 를 트리거하지만 eps 기본값만 다시 계산한다.

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
            if obj not in self.mesh_cache:
                mat = config.OBJECT_TARGET_MATERIAL.get(obj)  # 예: sample → 초록만. 미지정 시 전체 메시
                self.mesh_cache[obj] = load_meshes(obj, mat)
            full_mesh, target_mesh, input_path = self.mesh_cache[obj]

            sp = p["surface_spacing_mm"]
            gkey = surface_key(obj, p)
            if gkey not in self.surface_cache:
                fi = config.OBJECT_FILTER_INTERIOR.get(obj)  # hollow 물체만 opt-in
                self.surface_cache[gkey] = prepare_viewpoints(
                    target_mesh,
                    ViewpointGenParams(
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
            self.gen_status.content = (
                f"**Done** · {data['n']} vp · {ds['num_edges']} edges · "
                f"{ds['num_components']} component(s)")
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
            "method": "surface",
            "sampling_mode": "surface",
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
        try:
            save_viewpoints_hdf5(
                surface["positions"], surface["normals"], out, metadata, camera_spec,
                adjacency=L["adjacency"],
            )
            # 패널에는 짧은 이름만 — 절대경로를 코드 스팬으로 찍으면 마크다운이 '/' 에서
            # 줄바꿈을 못 해 패널이 가로로 넘친다(24em). 전체 경로는 콘솔에 남고, 어느
            # 파일인지는 바로 위 Saved viewpoints 드롭다운이 같은 라벨로 가리킨다.
            label = f"{out_path.parent.name}/{out_path.name}"
            self.gen_status.content = f"**Saved** → `{label}`"
            self._refresh_existing_options(select=label)
            print(f"[save] wrote {out}")
        except OSError as exc:
            self.gen_status.content = (
                f"**Save failed** ({exc.__class__.__name__})\n\n"
                f"디렉토리 권한 확인 (root 소유일 수 있음). 경로는 콘솔 로그에.")
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

        if full_mesh is not None:
            self.layers["mesh"].append(srv.scene.add_mesh_simple(
                "/scene/mesh",
                vertices=np.asarray(full_mesh.vertices), faces=np.asarray(full_mesh.faces),
                color=MESH_RGB, opacity=0.25, side="double"))
        else:
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

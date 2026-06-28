#!/usr/bin/env python3
"""Interactive viewpoint studio with viser.

Two ways to put viewpoints on screen, both object-centric:

  * **Generate** — pick an object, tune clustering parameters, and
    regenerate viewpoints in-process via the ``generate_viewpoints.py`` seam
    (``load_meshes`` / ``prepare_grid`` / ``cluster_coacd`` / ``cluster_and_order``).
    Viewpoints are generated with surface sampling only. Surface spacing is derived
    from camera FOV and overlap; CoACD is cached per (object, spacing, threshold)
    so tuning sub-cluster parameters is fast (~2s).
  * **Existing h5** — load a previously saved ``viewpoints*.h5`` for the object.

Rendered elements (same as the static plotly export, ``common/viewpoint_viz.py``):
translucent mesh, per-cluster markers, intra-cluster path lines, inter-cluster
transitions, and — for generated results — translucent CoACD part overlays.
Layers toggle independently; a playback slider scrubs/auto-plays the visit order.

Scope: sampling is fixed to ``surface`` and ordering to ``lawnmower`` in this app.
Grid sampling remains available in ``generate_viewpoints.py`` for CLI/batch use.
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
from common.viewpoint_viz import _BOLD_COLORS, _PART_COLORS
from core.generate_viewpoints import (
    load_meshes, prepare_grid as prepare_viewpoints, cluster_coacd, cluster_and_order,
    save_viewpoints_hdf5, ViewpointGenParams,
)

HIGHLIGHT_RGB = (255, 235, 59)   # moving playback marker
TRAIL_RGB = (255, 205, 0)        # visited path so far
TRANSITION_RGB = (150, 150, 150)  # inter-cluster lines
MESH_RGB = (180, 180, 180)
SURFACE_RGB = (255, 255, 255)

EPS_SPACING_FACTOR = 1.5  # dbscan 기본 eps = factor × FOV-derived spacing(mm)
DBSCAN_MIN_SAMPLES = 2
DBSCAN_NORMAL_WEIGHT = 0.0
OVERLAP_MIN_PCT = 20
OVERLAP_MAX_PCT = 90
SUBCLUSTER_METHODS = ["agglomerative", "dbscan"]
DEFAULT_SUBCLUSTER_METHOD = "agglomerative"

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


def fov_spacing_mm(overlap_pct: float) -> tuple[float, float, float]:
    """Return (row, col, isotropic_surface) spacing in mm from FOV and overlap."""
    overlap_ratio = _clamp(overlap_pct, OVERLAP_MIN_PCT, OVERLAP_MAX_PCT) / 100.0
    row_mm = config.CAMERA_FOV_HEIGHT_MM * (1.0 - overlap_ratio)
    col_mm = config.CAMERA_FOV_WIDTH_MM * (1.0 - overlap_ratio)
    return row_mm, col_mm, min(row_mm, col_mm)


def eps_default_mm(surface_spacing_mm: float) -> int:
    eps = EPS_SPACING_FACTOR * surface_spacing_mm
    eps = _clamp(eps, 5.0, 80.0)
    return int(eps + 0.5)


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


def _attr_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


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
    """Read positions/normals/clusters/path_order + mesh path + working distance."""
    with h5py.File(path, "r") as f:
        g = f["viewpoints"]
        positions = np.asarray(g["positions"], dtype=np.float64)
        normals = np.asarray(g["normals"], dtype=np.float64)
        n = len(positions)

        cluster_id = (np.asarray(g["cluster_id"], dtype=np.int32)
                      if "cluster_id" in g else np.zeros(n, dtype=np.int32))
        path_order = (np.asarray(g["path_order"], dtype=np.int32)
                      if "path_order" in g else np.arange(n, dtype=np.int32))
        cluster_order = (np.asarray(g["cluster_order"], dtype=np.int32)
                         if "cluster_order" in g else np.unique(cluster_id))

        wd_m = config.CAMERA_WORKING_DISTANCE_MM / 1000.0
        input_mesh = None
        if "metadata" in f:
            md = f["metadata"]
            if "input_mesh" in md.attrs:
                input_mesh = _attr_str(md.attrs["input_mesh"])
            if "camera_spec" in md and "working_distance_mm" in md["camera_spec"].attrs:
                wd_m = float(md["camera_spec"].attrs["working_distance_mm"]) / 1000.0

    camera_positions = positions + normals * wd_m
    return _scene_dict(positions, normals, camera_positions, cluster_id, cluster_order,
                       path_order, input_mesh, wd_m)


def _scene_dict(positions, normals, camera_positions, cluster_id, cluster_order,
                path_order, input_mesh, wd_m) -> dict:
    return {
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
            "mesh": [], "surface": [], "markers": [], "paths": [], "transitions": [], "coacd": [],
        }
        self.play: dict[str, object] = {"highlight": None, "visited": None}
        self.data: dict | None = None
        self.pb_pos = 0.0
        self.step_slider = None

        # caches (per object / per (object, surface spacing[, threshold]))
        self.mesh_cache: dict[str, tuple] = {}   # obj -> (full_mesh, target_mesh, input_path)
        self.surface_cache: dict[tuple, dict] = {}  # (obj, spacing) -> prepare_viewpoints result
        self.coacd_cache: dict[tuple, tuple] = {}  # (obj, spacing, threshold) -> (ids, parts)
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
            self.cb_mesh = g.add_checkbox("Mesh", initial_value=True)
            self.cb_surface = g.add_checkbox("Surface points", initial_value=True)
            self.cb_markers = g.add_checkbox("Markers", initial_value=True)
            self.cb_paths = g.add_checkbox("Cluster paths", initial_value=True)
            self.cb_transitions = g.add_checkbox("Transitions", initial_value=True)
            self.cb_coacd = g.add_checkbox("CoACD parts", initial_value=False)

        with g.add_folder("Generate (surface + coacd + sub-cluster)"):
            initial_overlap = default_overlap_pct()
            _, _, initial_spacing = fov_spacing_mm(initial_overlap)
            self.sl_overlap = g.add_slider(
                "FOV overlap (%)", min=OVERLAP_MIN_PCT, max=OVERLAP_MAX_PCT,
                step=1, initial_value=initial_overlap)
            self.fov_status = g.add_markdown("")
            self.sl_threshold = g.add_slider("coacd_threshold", min=0.05, max=0.5, step=0.05, initial_value=0.25)
            self.submethod_dd = g.add_dropdown(
                "Sub-cluster", options=SUBCLUSTER_METHODS, initial_value=DEFAULT_SUBCLUSTER_METHOD)
            # agglomerative 노브: 클러스터 최대 지름(mm). complete-linkage로 지름 ≤ 값 보장
            # → 멀리 떨어진 viewpoint가 한 클러스터로 묶이는 것 방지.
            self.sl_maxspan = g.add_slider("max span (mm)", min=50, max=500, step=10, initial_value=250)
            # dbscan 노브 (eps는 FOV-derived surface spacing을 자동 추적)
            self.sl_eps = g.add_slider(
                "eps (mm)", min=5, max=80, step=1,
                initial_value=eps_default_mm(initial_spacing))
            self.btn_generate = g.add_button("Generate")
            self.btn_save = g.add_button("Save h5")
            self.gen_status = g.add_markdown("Idle.")

        self.playback_folder = g.add_folder("Playback")
        with self.playback_folder:
            self.play_cb = g.add_checkbox("Play", initial_value=False)
            self.speed_slider = g.add_slider("Speed (vp/s)", min=1, max=60, step=1, initial_value=10)
        self._make_step_slider(1)

        self.info = g.add_markdown("Pick an object, then **Generate** — or choose an existing h5.")

        # callbacks
        self.object_dd.on_update(lambda _: self._on_object_change())
        self.existing_dd.on_update(lambda _: self._on_existing_change())
        for cb in (self.cb_mesh, self.cb_surface, self.cb_markers,
                   self.cb_paths, self.cb_transitions, self.cb_coacd):
            cb.on_update(lambda _: self._apply_visibility())
        self.btn_generate.on_click(lambda _: self._on_generate())
        self.btn_save.on_click(lambda _: self._on_save())
        self.submethod_dd.on_update(lambda _: self._apply_subcluster_visibility())
        self.sl_overlap.on_update(lambda _: self._on_overlap_change())
        self._apply_subcluster_visibility()
        self._refresh_fov_status()

    def _current_overlap_pct(self) -> float:
        return float(self.sl_overlap.value)

    def _current_spacing(self) -> tuple[float, float, float]:
        return fov_spacing_mm(self._current_overlap_pct())

    def _refresh_fov_status(self) -> None:
        row_mm, col_mm, surface_mm = self._current_spacing()
        self.fov_status.content = (
            f"FOV `{config.CAMERA_FOV_WIDTH_MM:.0f}×{config.CAMERA_FOV_HEIGHT_MM:.0f} mm` · "
            f"overlap `{self._current_overlap_pct():.0f}%` · "
            f"surface spacing `{surface_mm:.1f} mm` "
            f"(row `{row_mm:.1f}`, col `{col_mm:.1f}`)"
        )

    def _on_overlap_change(self) -> None:
        """FOV overlap이 바뀌면 surface spacing과 dbscan eps 기본값을 같이 갱신한다."""
        _, _, surface_mm = self._current_spacing()
        self.sl_eps.value = eps_default_mm(surface_mm)
        self._refresh_fov_status()

    def _apply_subcluster_visibility(self) -> None:
        """Show only controls relevant to the selected sub-clustering method."""
        method = str(self.submethod_dd.value)
        is_agglomerative = method == "agglomerative"
        is_dbscan = method == "dbscan"

        self.sl_maxspan.visible = is_agglomerative
        self.sl_eps.visible = is_dbscan

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
        self._set_scene(full, data, coacd_parts=None, source=f"h5: {label}")

    def _on_generate(self) -> None:
        if self.generating:
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
        method = f"coacd+{submethod}"
        row_spacing_mm, col_spacing_mm, surface_spacing_mm = self._current_spacing()
        p = {
            "obj": self.object_dd.value,
            "sampling_mode": "surface",
            "ordering_mode": "lawnmower",
            "surface_overlap_pct": self._current_overlap_pct(),
            "surface_spacing_mm": surface_spacing_mm,
            "row_spacing_mm": row_spacing_mm,
            "col_spacing_mm": col_spacing_mm,
            "submethod": submethod,
            "method": method,
            "threshold": float(self.sl_threshold.value),
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
            gkey = (obj, round(sp, 4))
            if gkey not in self.surface_cache:
                self.surface_cache[gkey] = prepare_viewpoints(
                    target_mesh,
                    ViewpointGenParams(
                        sampling_mode="surface",
                        ordering_mode="lawnmower",
                        surface_spacing_mm=sp,
                        row_spacing_mm=p["row_spacing_mm"],
                        col_spacing_mm=p["col_spacing_mm"],
                    ),
                )
            surface = self.surface_cache[gkey]

            ckey = (obj, round(sp, 4), round(p["threshold"], 4))
            if ckey not in self.coacd_cache:
                self.coacd_cache[ckey] = cluster_coacd(target_mesh, surface["positions"], p["threshold"])
            cached = self.coacd_cache[ckey]

            method = p["method"]  # coacd+agglomerative | coacd+dbscan
            common = dict(
                positions=surface["positions"], normals=surface["normals"],
                camera_positions=surface["camera_positions"], target_mesh=target_mesh,
                row_spacing_m=surface["row_spacing_m"], col_spacing_m=surface["col_spacing_m"],
                grid_row_index=surface["grid_row_index"],
                cam_axis1=surface["cam_axis1"], cam_axis2=surface["cam_axis2"],
                original_path_length_mm=surface["original_path_length_mm"],
                threshold=p["threshold"], normal_weight=p["normal_weight"],
                precomputed_coacd=cached, ordering_mode=p["ordering_mode"],
            )
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
                str(input_path), config.CAMERA_WORKING_DISTANCE_MM / 1000.0,
            )
            self.last = {"obj": obj, "surface": surface, "result": result,
                         "params": p, "n": data["n"], "input_path": input_path}
            red = (1 - result["path_length_mm"] / surface["original_path_length_mm"]) * 100
            smlabel = f"surface {p['surface_overlap_pct']:.0f}% overlap · {sp:.1f}mm"
            if p["submethod"] == "agglomerative":
                knob = f"span={p['max_span_mm']:.0f}mm"
            else:
                knob = f"eps={p['eps_mm']:.0f}mm"
            self._set_scene(
                full_mesh, data, coacd_parts=result.get("coacd_parts"),
                source=f"gen · {smlabel} · {method} · t={p['threshold']} {knob}",
            )
            self.gen_status.content = (
                f"**Done** · {smlabel} · {method} ({knob}) · {data['n']} vp · "
                f"{result['num_clusters']} clusters · "
                f"{len(result.get('coacd_parts') or [])} CoACD parts · "
                f"path {result['path_length_mm']:.0f} mm ({red:.1f}% reduction)")
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
        clmethod = p.get("method", "coacd+agglomerative")
        out = str(config.get_viewpoint_path(obj, L["n"], filename=f"viewpoints_{clmethod}.h5"))
        camera_spec = {
            "fov_width_mm": config.CAMERA_FOV_WIDTH_MM,
            "fov_height_mm": config.CAMERA_FOV_HEIGHT_MM,
            "working_distance_mm": config.CAMERA_WORKING_DISTANCE_MM,
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
            "coacd_threshold": p["threshold"],
        }
        if p.get("submethod") == "agglomerative":
            cluster_meta["max_span_mm"] = p["max_span_mm"]
        elif p.get("submethod") == "dbscan":
            cluster_meta["dbscan_eps_mm"] = p["eps_mm"]
            cluster_meta["dbscan_min_samples"] = p["min_samples"]
            cluster_meta["dbscan_normal_weight"] = p["normal_weight"]
        pca_data = {
            "center": surface["pca_center"],
            "axis1": surface["pca_axis1"],
            "axis2": surface["pca_axis2"],
        }
        try:
            save_viewpoints_hdf5(
                surface["positions"], surface["normals"], out, metadata, camera_spec,
                result["path_order"], pca_data, surface["row_index"],
                cluster_id=result["cluster_ids"], cluster_order=result["cluster_order"],
                cluster_direction=result["cluster_direction"], cluster_metadata=cluster_meta,
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
            "transitions": self.cb_transitions, "coacd": self.cb_coacd,
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

        if full_mesh is not None:
            self.layers["mesh"].append(srv.scene.add_mesh_simple(
                "/scene/mesh",
                vertices=np.asarray(full_mesh.vertices), faces=np.asarray(full_mesh.faces),
                color=MESH_RGB, opacity=0.25, side="double"))
        else:
            print("  [warn] no mesh to display; skipping mesh layer")

        palette = distinct_colors(len(corder))  # K개 고유 색 (재사용 없음)
        for rank, c in enumerate(corder):
            idx = np.where(cid == c)[0]
            if idx.size == 0:
                continue
            rgb = palette[rank]
            self.layers["surface"].append(srv.scene.add_point_cloud(
                f"/scene/surface/c{c}", points=surf[idx],
                colors=np.tile(np.array(rgb, dtype=np.uint8), (len(idx), 1)),
                point_size=0.0025, point_shape="circle"))
            self.layers["markers"].append(srv.scene.add_point_cloud(
                f"/scene/markers/c{c}", points=cam[idx],
                colors=np.tile(np.array(rgb, dtype=np.uint8), (len(idx), 1)),
                point_size=0.004, point_shape="circle"))
            ordered = idx[np.argsort(porder[idx], kind="stable")]
            if ordered.size > 1:
                self.layers["paths"].append(srv.scene.add_spline_catmull_rom(
                    f"/scene/paths/c{c}", positions=cam[ordered],
                    color=rgb, line_width=3.0, curve_type="catmullrom"))

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

        if coacd_parts:
            for j, part in enumerate(coacd_parts):
                self.layers["coacd"].append(srv.scene.add_mesh_simple(
                    f"/scene/coacd/p{j}",
                    vertices=np.asarray(part.vertices), faces=np.asarray(part.faces),
                    color=part_rgb(j), opacity=0.3, side="double"))

        self._apply_visibility()

    def _set_scene(self, full_mesh, data: dict, coacd_parts, source: str) -> None:
        self.data = data
        self._build_scene(full_mesh, data, coacd_parts)
        self._make_step_slider(data["n"])
        self.pb_pos = 0.0
        self._update_highlight(0)
        self.info.content = "\n".join([
            f"**Source:** `{source}`",
            f"**Viewpoints:** `{data['n']}`",
            f"**Clusters:** `{len(data['cluster_order'])}`",
            f"**Working dist:** `{data['wd_m'] * 1000:.0f} mm`",
        ])
        print(f"Scene: {source} ({data['n']} vp, {len(data['cluster_order'])} clusters)")

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

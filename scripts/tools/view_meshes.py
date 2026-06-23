#!/usr/bin/env python3
"""Browse pipeline object meshes with viser.

By default this scans `data/*/mesh/source.obj` and lets you switch objects from
the viser control panel. Use `--all-obj` to include every OBJ under each mesh
directory, including raw FreeCAD exports.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import viser


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class MeshEntry:
    label: str
    path: Path


def discover_meshes(data_root: Path, include_all_obj: bool) -> list[MeshEntry]:
    pattern = "*/mesh/*.obj" if include_all_obj else "*/mesh/source.obj"
    entries: list[MeshEntry] = []

    for path in sorted(data_root.glob(pattern)):
        object_name = path.parent.parent.name
        label = object_name if path.name == "source.obj" else f"{object_name}/{path.name}"
        entries.append(MeshEntry(label=label, path=path.resolve()))

    return entries


def load_as_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"No geometry found in {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)!r}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"Mesh has no vertices or faces: {path}")
    return loaded


def format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.6g}" for v in values) + "]"


def mesh_info_markdown(label: str, path: Path, mesh: trimesh.Trimesh) -> str:
    rel_path = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    return "\n".join(
        [
            f"**Object:** `{label}`",
            f"**File:** `{rel_path}`",
            f"**Vertices:** `{len(mesh.vertices):,}`",
            f"**Faces:** `{len(mesh.faces):,}`",
            f"**Extents:** `{format_vector(mesh.extents)}`",
            f"**Bounds min:** `{format_vector(mesh.bounds[0])}`",
            f"**Bounds max:** `{format_vector(mesh.bounds[1])}`",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a browser UI for selecting and visualizing object meshes."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing data/{object}/mesh/source.obj entries.",
    )
    parser.add_argument(
        "--all-obj",
        action="store_true",
        help="List every OBJ under data/*/mesh instead of source.obj only.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Extra display scale. Keep 1.0 for normalized source.obj files.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center each mesh bbox in the viewer for inspection only.",
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="Show a translucent wireframe overlay.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    entries = discover_meshes(data_root, args.all_obj)
    if not entries:
        raise SystemExit(f"No OBJ meshes found under {data_root}")

    entry_by_label = {entry.label: entry for entry in entries}
    current_handles: list[object] = []

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(
        control_layout="collapsible",
        control_width="medium",
        dark_mode=True,
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=1.0,
        height=1.0,
        plane="xy",
        cell_size=0.05,
        section_size=0.25,
    )
    server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.004)

    dropdown = server.gui.add_dropdown(
        "Object",
        options=[entry.label for entry in entries],
        initial_value=entries[0].label,
    )
    wireframe_checkbox = server.gui.add_checkbox(
        "Wireframe",
        initial_value=args.wireframe,
    )
    info = server.gui.add_markdown("Loading...")

    def clear_current_mesh() -> None:
        while current_handles:
            handle = current_handles.pop()
            handle.remove()

    def add_selected_mesh() -> None:
        label = dropdown.value
        entry = entry_by_label[label]
        mesh = load_as_trimesh(entry.path)

        if args.scale != 1.0:
            mesh.apply_scale(args.scale)
        if args.center:
            mesh.apply_translation(-mesh.bounding_box.centroid)

        clear_current_mesh()
        current_handles.append(server.scene.add_mesh_trimesh("/mesh", mesh=mesh))

        if wireframe_checkbox.value:
            current_handles.append(
                server.scene.add_mesh_simple(
                    "/mesh_wireframe",
                    vertices=np.asarray(mesh.vertices),
                    faces=np.asarray(mesh.faces),
                    color=(20, 20, 20),
                    wireframe=True,
                    opacity=0.35,
                    side="double",
                )
            )

        max_extent = float(np.max(mesh.extents))
        current_handles.append(
            server.scene.add_frame(
                "/mesh_frame",
                axes_length=max(0.05, max_extent * 0.35),
                axes_radius=max(0.002, max_extent * 0.01),
            )
        )

        info.content = mesh_info_markdown(label, entry.path, mesh)
        print(f"Loaded {label}: {entry.path}")
        print(f"  extents: {mesh.extents}")
        print(f"  bounds:  {mesh.bounds}")

    @dropdown.on_update
    def _(_: viser.GuiEvent) -> None:
        add_selected_mesh()

    @wireframe_checkbox.on_update
    def _(_: viser.GuiEvent) -> None:
        add_selected_mesh()

    add_selected_mesh()

    print("Available objects:")
    for entry in entries:
        rel_path = entry.path.relative_to(PROJECT_ROOT) if entry.path.is_relative_to(PROJECT_ROOT) else entry.path
        print(f"  {entry.label}: {rel_path}")
    print(f"Open: http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

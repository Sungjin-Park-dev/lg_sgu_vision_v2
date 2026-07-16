"""GLNS text-tour and result-HDF5 persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np

from .problem import RESULT_FORMAT_VERSION

def write_simple_gtsp(path: Path, problem: dict) -> None:
    """Write the compact GLNS ``N:/M:/sets/matrix`` input format."""
    costs = np.asarray(problem["costs"], dtype=np.int64)
    sets = problem["sets"]
    lines = [f"N: {len(costs)}", f"M: {len(sets)}"]
    for sid, vertices in enumerate(sets, start=1):
        one_based = " ".join(str(int(v) + 1) for v in np.asarray(vertices))
        lines.append(f"{sid} {one_based}")
    lines.extend(" ".join(str(int(v)) for v in row) for row in costs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_glns_tour(path: Path) -> np.ndarray:
    """Parse GLNS's ``Tour : [1, 2, ...]`` output into zero-based IDs."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Tour\s*:\s*\[([^\]]*)\]", text)
    if match is None:
        raise ValueError(f"GLNS output has no Tour field: {path}")
    values = [int(token) - 1 for token in re.findall(r"-?\d+", match.group(1))]
    return np.asarray(values, dtype=np.int32)


def decode_and_validate_tour(tour: np.ndarray, problem: dict) -> dict:
    """Remove the dummy, map vertices to viewpoint/IK choices, and validate."""
    tour = np.asarray(tour, dtype=np.int32)
    costs = np.asarray(problem["costs"], dtype=np.int64)
    n_sets = len(problem["sets"])
    if len(tour) != n_sets or len(np.unique(tour)) != len(tour):
        raise ValueError(f"tour must select exactly one vertex from each of {n_sets} sets")
    if np.any(tour < 0) or np.any(tour >= len(costs)):
        raise ValueError("tour contains an out-of-range vertex")

    membership = np.full(len(costs), -1, dtype=np.int32)
    for sid, vertices in enumerate(problem["sets"]):
        membership[np.asarray(vertices, dtype=np.int32)] = sid
    chosen_sets = membership[tour]
    if set(int(x) for x in chosen_sets) != set(range(n_sets)):
        raise ValueError("tour does not choose one vertex from every GTSP set")

    dummy = int(problem["dummy_vertex"])
    dummy_positions = np.where(tour == dummy)[0]
    if len(dummy_positions) != 1:
        raise ValueError("tour must contain the dummy vertex exactly once")
    cut = int(dummy_positions[0])
    ordered_vertices = np.concatenate([tour[cut + 1:], tour[:cut]])
    viewpoint = problem["vertex_viewpoint"][ordered_vertices]
    candidate = problem["vertex_candidate"][ordered_vertices]

    allowed = problem["allowed_view_edges"]
    for a, b in zip(viewpoint[:-1], viewpoint[1:]):
        edge = (min(int(a), int(b)), max(int(a), int(b)))
        if edge not in allowed:
            raise ValueError(f"GLNS selected forbidden non-Delaunay transition {edge}")

    cycle = np.concatenate([tour, tour[:1]])
    cycle_costs = costs[cycle[:-1], cycle[1:]]
    if np.any(cycle_costs >= int(problem["forbidden_cost"])):
        raise ValueError("GLNS tour contains a forbidden-cost edge")
    return {
        "vertices": ordered_vertices.astype(np.int32),
        "viewpoint_order": viewpoint.astype(np.int32),
        "candidate_order": candidate.astype(np.int32),
        "cost": int(cycle_costs.sum()),
    }


def write_result_hdf5(
    output_path: Path,
    metadata: dict,
    reachable_mask: np.ndarray,
    candidate_counts: np.ndarray,
    induced_edges: np.ndarray,
    component_id: np.ndarray,
    components: list[dict],
    candidate_counts_raw: np.ndarray | None = None,
) -> Path:
    """Write the standalone GLNS experiment result (not a viewpoint file)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["format"] = "delaunay_glns_result"
        f.attrs["format_version"] = RESULT_FORMAT_VERSION
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(value)
            else:
                f.attrs[key] = value

        input_group = f.create_group("input")
        input_group.create_dataset("reachable_mask", data=np.asarray(reachable_mask, dtype=bool))
        input_group.create_dataset("candidate_counts", data=np.asarray(candidate_counts, dtype=np.int32))
        if candidate_counts_raw is not None:
            input_group.create_dataset(
                "candidate_counts_raw", data=np.asarray(candidate_counts_raw, dtype=np.int32),
            )
        input_group.create_dataset("induced_delaunay_edges", data=np.asarray(induced_edges, dtype=np.int32))
        input_group.create_dataset("component_id", data=np.asarray(component_id, dtype=np.int32))

        components_group = f.create_group("components")
        for cid, component in enumerate(components):
            group = components_group.create_group(f"{cid:03d}")
            group.attrs["status"] = str(component["status"])
            group.attrs["reason"] = str(component.get("reason", ""))
            for key in (
                "solver_cost", "reconfig_unit", "reconfig_unit_base", "reconfig_unit_any",
                "reconfig_unit_wrist", "tilt_unit", "forbidden_cost", "joint_cost_scale",
                "objective_base_cost", "objective_any_cost", "objective_tilt_cost",
                "objective_joint_cost", "num_reconfigurations_any",
                "num_reconfigurations", "num_reconfigurations_base",
                "num_reconfigurations_wrist", "solver_seconds", "matrix_mib",
            ):
                if key in component and component[key] is not None:
                    group.attrs[key] = component[key]
            group.create_dataset("members", data=np.asarray(component["members"], dtype=np.int32))
            for key in ("candidate_counts", "candidate_counts_raw"):
                if component.get(key) is not None:
                    group.create_dataset(key, data=np.asarray(component[key], dtype=np.int32))
            if component.get("feasibility_witness") is not None:
                group.create_dataset(
                    "feasibility_witness",
                    data=np.asarray(component["feasibility_witness"], dtype=np.int32),
                )
            if component.get("viewpoint_order") is not None:
                for key, dtype in (
                    ("viewpoint_order", np.int32),
                    ("selected_candidate_index", np.int32),
                    ("selected_joints", np.float64),
                    ("selected_joint_turns", np.int16),
                    ("edge_linf_rad", np.float64),
                    ("edge_linf_base_rad", np.float64),
                    ("edge_linf_wrist_rad", np.float64),
                    ("edge_l2_rad", np.float64),
                    ("is_reconfiguration", bool),
                    ("is_reconfiguration_base", bool),
                    ("is_reconfiguration_wrist", bool),
                    ("selected_pose_variant", "S16"),
                    ("selected_roll_deg", np.float64),
                    ("selected_tilt_deg", np.float64),
                    ("selected_tilt_azimuth_deg", np.float64),
                    ("selected_target_position", np.float64),
                    ("selected_target_quaternion", np.float64),
                ):
                    value = component.get(key)
                    if value is None:
                        continue  # tolerate older/partial component dicts
                    group.create_dataset(key, data=np.asarray(value, dtype=dtype))
    return output_path


def read_result_hdf5(path: Path) -> dict:
    """Load a GLNS result for the Viser inspector and tests."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != "delaunay_glns_result":
            raise ValueError(f"not a Delaunay GLNS result: {path}")
        metadata = {key: f.attrs[key] for key in f.attrs}
        input_group = f["input"]
        result = {
            "path": path,
            "metadata": metadata,
            "reachable_mask": np.asarray(input_group["reachable_mask"], dtype=bool),
            "candidate_counts": np.asarray(input_group["candidate_counts"], dtype=np.int32),
            "candidate_counts_raw": np.asarray(
                input_group.get("candidate_counts_raw", input_group["candidate_counts"]),
                dtype=np.int32,
            ),
            "induced_edges": np.asarray(input_group["induced_delaunay_edges"], dtype=np.int32),
            "component_id": np.asarray(input_group["component_id"], dtype=np.int32),
            "components": [],
        }
        for name in sorted(f["components"]):
            group = f["components"][name]
            component = {
                "name": name,
                "status": str(group.attrs["status"]),
                "reason": str(group.attrs.get("reason", "")),
                "attrs": {key: group.attrs[key] for key in group.attrs},
                "members": np.asarray(group["members"], dtype=np.int32),
            }
            for key in (
                "feasibility_witness", "viewpoint_order", "selected_candidate_index",
                "selected_joints", "edge_linf_rad", "edge_linf_base_rad",
                "selected_joint_turns",
                "edge_linf_wrist_rad", "edge_l2_rad", "is_reconfiguration",
                "is_reconfiguration_base", "is_reconfiguration_wrist",
                "selected_pose_variant", "selected_roll_deg", "selected_tilt_deg",
                "selected_tilt_azimuth_deg", "selected_target_position",
                "selected_target_quaternion",
                "candidate_counts", "candidate_counts_raw",
            ):
                component[key] = np.asarray(group[key]) if key in group else None
            result["components"].append(component)
    return result

"""Pure-Python helpers for Delaunay-constrained GLNS experiments.

This module deliberately has no cuRobo dependency.  It owns graph induction,
Hamiltonian-path feasibility checks, GTSP matrix construction, GLNS text I/O,
and the result-HDF5 schema so those pieces can be unit-tested without a GPU.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from ortools.sat.python import cp_model
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


RESULT_FORMAT_VERSION = 2
JOINT_COST_SCALE = 1000


def effective_candidate_cap(
    component_size: int,
    requested_cap: int = 16,
    matrix_target_mib: float = 256.0,
) -> int:
    """Largest uniform per-viewpoint cap fitting the target dense Int64 matrix."""
    if component_size <= 0 or requested_cap <= 0 or matrix_target_mib <= 0.0:
        raise ValueError("component_size, requested_cap and matrix_target_mib must be positive")
    matrix_bytes = float(matrix_target_mib) * 1024.0 ** 2
    cap = int(np.floor((np.sqrt(matrix_bytes / 8.0) - 1.0) / component_size))
    return max(1, min(int(requested_cap), cap))


def prune_candidate_sets(
    representatives: list[np.ndarray],
    metadata: list[dict[str, np.ndarray]],
    edges: np.ndarray,
    cap_by_viewpoint: np.ndarray,
    threshold_rad: float,
    joint_weights: np.ndarray,
    reference_joints: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Deterministically retain connected, diverse IK candidates per viewpoint.

    Nominal candidates are selected before augmented candidates. Within each
    pool the greedy ordering is incident base-connectivity, incident full-6D
    connectivity, lower tilt, greater distance from already selected branches,
    then lower weighted distance to the reference configuration.
    """
    n = len(representatives)
    if len(metadata) != n or np.asarray(cap_by_viewpoint).shape != (n,):
        raise ValueError("candidate arrays and cap_by_viewpoint must have matching lengths")
    weights = np.asarray(joint_weights, dtype=np.float64)
    ref = np.zeros(6) if reference_joints is None else np.asarray(reference_joints, dtype=np.float64)
    if weights.shape != (6,) or ref.shape != (6,):
        raise ValueError("joint_weights and reference_joints must have shape (6,)")

    incident: list[list[int]] = [[] for _ in range(n)]
    for a, b in np.asarray(edges, dtype=np.int32):
        incident[int(a)].append(int(b))
        incident[int(b)].append(int(a))

    out_reps: list[np.ndarray] = []
    out_meta: list[dict[str, np.ndarray]] = []
    for vp in range(n):
        reps = np.asarray(representatives[vp], dtype=np.float64)
        md = metadata[vp]
        k = len(reps)
        cap = min(k, max(1, int(cap_by_viewpoint[vp])))
        if k <= cap:
            chosen = list(range(k))
        else:
            base_conn = np.zeros(k, dtype=np.int32)
            any_conn = np.zeros(k, dtype=np.int32)
            for nb in sorted(set(incident[vp])):
                other = np.asarray(representatives[nb], dtype=np.float64)
                if not len(other):
                    continue
                delta = np.abs(reps[:, None, :] - other[None, :, :])
                base_conn += np.any(np.max(delta[..., :3], axis=2) <= threshold_rad, axis=1)
                any_conn += np.any(np.max(delta, axis=2) <= threshold_rad, axis=1)
            variants = np.asarray(md["variant"]).astype("U")
            tilt = np.asarray(md["tilt_deg"], dtype=np.float64)
            distance = np.linalg.norm((reps - ref) * weights, axis=1)
            chosen: list[int] = []

            def choose_from(pool: list[int], limit: int) -> None:
                remaining = list(pool)
                while remaining and len(chosen) < limit:
                    def key(i: int):
                        diversity = (np.inf if not chosen else
                                     min(np.linalg.norm((reps[i] - reps[j]) * weights)
                                         for j in chosen))
                        return (-int(base_conn[i]), -int(any_conn[i]), float(tilt[i]),
                                -float(diversity), float(distance[i]), int(i))
                    best = min(remaining, key=key)
                    chosen.append(best)
                    remaining.remove(best)

            nominal = [i for i in range(k) if variants[i] == "nominal"]
            augmented = [i for i in range(k) if variants[i] != "nominal"]
            choose_from(nominal, cap)
            choose_from(augmented, cap)

        idx = np.asarray(chosen, dtype=np.int32)
        out_reps.append(reps[idx])
        out_meta.append({key: np.asarray(value)[idx] for key, value in md.items()})
    return out_reps, out_meta


def induce_adjacency(
    edges: np.ndarray,
    keep_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return the original-index induced graph and recomputed components.

    ``component_id`` has source-viewpoint length. Dropped vertices are ``-1``;
    retained vertices use contiguous component IDs starting at zero.
    """
    keep = np.asarray(keep_mask, dtype=bool)
    source_edges = np.asarray(edges, dtype=np.int32)
    if source_edges.ndim != 2 or source_edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got {source_edges.shape}")
    n = len(keep)
    if len(source_edges):
        if np.any(source_edges < 0) or np.any(source_edges >= n):
            raise ValueError("adjacency edge index out of range")
        edge_keep = keep[source_edges[:, 0]] & keep[source_edges[:, 1]]
        induced = source_edges[edge_keep]
    else:
        induced = np.empty((0, 2), dtype=np.int32)

    retained = np.flatnonzero(keep).astype(np.int32)
    component_id = np.full(n, -1, dtype=np.int32)
    if len(retained) == 0:
        return induced, component_id, []

    local_of = np.full(n, -1, dtype=np.int32)
    local_of[retained] = np.arange(len(retained), dtype=np.int32)
    if len(induced):
        local_edges = local_of[induced]
        rows = np.concatenate([local_edges[:, 0], local_edges[:, 1]])
        cols = np.concatenate([local_edges[:, 1], local_edges[:, 0]])
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(len(retained), len(retained)),
        )
        n_components, local_labels = connected_components(graph, directed=False)
    else:
        n_components = len(retained)
        local_labels = np.arange(len(retained), dtype=np.int32)

    component_id[retained] = np.asarray(local_labels, dtype=np.int32)
    components = [
        retained[local_labels == cid].astype(np.int32)
        for cid in range(int(n_components))
    ]
    return induced.astype(np.int32), component_id, components


def expand_edges_by_hops(edges: np.ndarray, n_nodes: int, hops: int) -> np.ndarray:
    """Relax the Delaunay graph to all node pairs within ``hops`` hops.

    ``hops=1`` returns the input edge set unchanged (canonicalized, undirected).
    ``hops>=2`` adds every pair of nodes joined by a Delaunay path of length
    ``<= hops``, giving GLNS more routing freedom than strict Delaunay adjacency
    while staying on the surface graph (unlike a raw 3D k-NN, this never bridges
    geometrically-close-but-topologically-far patches). Returned shape is (E, 2).
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    edges = np.asarray(edges, dtype=np.int64)
    base = canonical_edge_set(edges) if len(edges) else set()
    if hops == 1 or not base:
        return np.asarray(sorted(base), dtype=np.int32).reshape(-1, 2)

    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, b in base:
        adjacency[a].append(b)
        adjacency[b].append(a)

    result = set(base)
    for src in range(n_nodes):
        visited = {src}
        frontier = {src}
        for _ in range(hops):
            nxt: set[int] = set()
            for u in frontier:
                for v in adjacency[u]:
                    if v not in visited:
                        nxt.add(v)
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        for dst in visited:
            if dst != src:
                result.add((min(src, dst), max(src, dst)))
    return np.asarray(sorted(result), dtype=np.int32).reshape(-1, 2)


def find_hamiltonian_open_path(
    members: np.ndarray,
    edges: np.ndarray,
    timeout_s: float = 5.0,
) -> tuple[str, np.ndarray | None]:
    """Check Delaunay-only open Hamiltonian feasibility with CP-SAT.

    A dummy node connected to every real node converts an open path into a
    circuit. The returned witness is only a feasibility diagnostic; GLNS still
    chooses the final viewpoint order and IK candidates.
    """
    nodes = np.asarray(members, dtype=np.int32)
    if len(nodes) == 0:
        return "empty", np.empty((0,), dtype=np.int32)
    if len(nodes) == 1:
        return "trivial", nodes.copy()

    node_set = set(int(x) for x in nodes)
    local_of = {int(source): local for local, source in enumerate(nodes)}
    local_edges = [
        (local_of[int(a)], local_of[int(b)])
        for a, b in np.asarray(edges, dtype=np.int32)
        if int(a) in node_set and int(b) in node_set
    ]

    model = cp_model.CpModel()
    arcs: list[tuple[int, int, cp_model.IntVar]] = []
    for a, b in local_edges:
        arcs.append((a, b, model.NewBoolVar(f"edge_{a}_{b}")))
        arcs.append((b, a, model.NewBoolVar(f"edge_{b}_{a}")))
    dummy = len(nodes)
    for i in range(len(nodes)):
        arcs.append((dummy, i, model.NewBoolVar(f"dummy_{i}")))
        arcs.append((i, dummy, model.NewBoolVar(f"{i}_dummy")))
    model.AddCircuit(arcs)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout_s)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    status_name = solver.StatusName(status).lower()
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return status_name, None

    successor: dict[int, int] = {}
    for a, b, variable in arcs:
        if solver.BooleanValue(variable):
            successor[a] = b
    witness_local = []
    current = successor[dummy]
    while current != dummy:
        if current in witness_local:
            raise RuntimeError("CP-SAT returned a malformed Hamiltonian circuit")
        witness_local.append(current)
        current = successor[current]
    witness = nodes[np.asarray(witness_local, dtype=np.int32)]
    if len(witness) != len(nodes):
        raise RuntimeError("CP-SAT witness does not cover every component member")
    return status_name, witness.astype(np.int32)


def build_gtsp_problem(
    members: np.ndarray,
    representatives: list[np.ndarray],
    edges: np.ndarray,
    reconfig_threshold_rad: float,
    joint_cost_scale: int = JOINT_COST_SCALE,
    joint_weights: np.ndarray | None = None,
    reconfig_base_joints: tuple[int, ...] = (0, 1, 2),
    reconfig_wrist_joints: tuple[int, ...] = (3, 4, 5),
    reconfig_weight_base: float = 12.0,
    reconfig_weight_wrist: float = 1.0,
    reconfig_exclude_last: bool = False,
    candidate_tilt_costs: list[np.ndarray] | None = None,
) -> dict:
    """Build a complete integer GTSP matrix with Delaunay non-edges forbidden.

    Every source viewpoint is one GTSP set and each collision-free IK
    representative is a vertex in that set. A final singleton dummy set opens
    the otherwise cyclic GLNS tour.

    The integer objective is strict lexicographic, in this order: number of
    base-joint (q0:q3) reconfigurations, number of any-six-joint
    reconfigurations, accumulated vertex tilt cost, weighted joint L2 travel.
    Tilt vertex costs are placed on both incident symmetric edges, including
    dummy endpoint edges, so every selected pose contributes exactly twice.

    The legacy reconfiguration arguments remain accepted for API compatibility;
    the strict tiers deliberately use q0:q3 and q0:q6 to match the verifier.
    """
    members = np.asarray(members, dtype=np.int32)
    if len(members) < 2:
        raise ValueError("GTSP construction requires at least two viewpoints")
    if reconfig_threshold_rad <= 0.0:
        raise ValueError("reconfig_threshold_rad must be > 0")
    if joint_cost_scale <= 0:
        raise ValueError("joint_cost_scale must be > 0")

    weights = (np.ones(6, dtype=np.float64) if joint_weights is None
               else np.asarray(joint_weights, dtype=np.float64))
    if weights.shape != (6,) or np.any(weights < 0.0):
        raise ValueError("joint_weights must be 6 non-negative values")
    base_idx = np.arange(3, dtype=np.int32)
    any_idx = np.arange(6, dtype=np.int32)

    member_set = set(int(x) for x in members)
    allowed_view_edges = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in np.asarray(edges, dtype=np.int32)
        if int(a) in member_set and int(b) in member_set
    }

    sets: list[np.ndarray] = []
    vertex_viewpoint: list[int] = []
    vertex_candidate: list[int] = []
    ranges: dict[int, np.ndarray] = {}
    tilt_by_vertex: list[int] = []
    for viewpoint in members:
        vp = int(viewpoint)
        reps = np.asarray(representatives[vp], dtype=np.float64)
        if reps.ndim != 2 or reps.shape[1] != 6 or len(reps) == 0:
            raise ValueError(f"viewpoint {vp} has no valid (K, 6) representatives")
        ids = np.arange(len(vertex_viewpoint), len(vertex_viewpoint) + len(reps), dtype=np.int32)
        ranges[vp] = ids
        sets.append(ids)
        vertex_viewpoint.extend([vp] * len(reps))
        vertex_candidate.extend(range(len(reps)))
        if candidate_tilt_costs is None:
            tilt_by_vertex.extend([0] * len(reps))
        else:
            tc = np.asarray(candidate_tilt_costs[vp], dtype=np.int64)
            if tc.shape != (len(reps),) or np.any(tc < 0):
                raise ValueError(f"viewpoint {vp} tilt costs must have shape ({len(reps)},)")
            tilt_by_vertex.extend(int(x) for x in tc)

    dummy_vertex = len(vertex_viewpoint)
    vertex_viewpoint.append(-1)
    vertex_candidate.append(-1)
    sets.append(np.array([dummy_vertex], dtype=np.int32))
    n_vertices = dummy_vertex + 1

    # Compute every allowed IK-pair secondary cost + per-tier L-inf jumps before
    # selecting the exact lexicographic penalties.
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    max_joint_cost = 0
    for a, b in sorted(allowed_view_edges):
        qa = np.asarray(representatives[a], dtype=np.float64)
        qb = np.asarray(representatives[b], dtype=np.float64)
        diff = qa[:, None, :] - qb[None, :, :]
        # 2차(동점 깨기) 비용은 per-joint 가중 L2 — wrist roll 등 싼 관절은 작게 반영.
        l2 = np.linalg.norm(diff * weights[None, None, :], axis=2)
        joint_cost = np.rint(l2 * joint_cost_scale).astype(np.int64)
        if joint_cost.size:
            max_joint_cost = max(max_joint_cost, int(joint_cost.max()))
        absd = np.abs(diff)
        # Strict tiers use base q0:q3 and all joints q0:q6.
        linf_base = np.max(absd[..., base_idx], axis=2)
        linf_any = np.max(absd[..., any_idx], axis=2)
        blocks.append((ranges[a], ranges[b], joint_cost, linf_base, linf_any))

    # Exact tier bounds. Each unit is one larger than the maximum total cost of
    # all lower tiers, therefore no amount of lower-tier improvement can trade
    # against one unit in a higher tier.
    n_real_edges = len(members) - 1
    max_l2_sum = n_real_edges * max_joint_cost
    max_tilt_vertex = max(tilt_by_vertex, default=0)
    tilt_unit = max_l2_sum + 1
    max_tilt_sum = 2 * len(members) * max_tilt_vertex * tilt_unit
    reconfig_unit_any = max_tilt_sum + max_l2_sum + 1
    max_any_sum = n_real_edges * reconfig_unit_any
    reconfig_unit_base = max_any_sum + max_tilt_sum + max_l2_sum + 1
    max_allowed_tour = (
        n_real_edges * reconfig_unit_base + max_any_sum
        + max_tilt_sum + max_l2_sum
    )
    forbidden_cost = max_allowed_tour + 1
    if forbidden_cost > np.iinfo(np.int64).max:
        raise OverflowError("strict lexicographic GTSP costs exceed Int64")
    costs = np.full((n_vertices, n_vertices), forbidden_cost, dtype=np.int64)
    tilt_arr = np.asarray(tilt_by_vertex, dtype=np.int64)

    for ids_a, ids_b, joint_cost, linf_base, linf_any in blocks:
        edge_tilt = tilt_arr[ids_a, None] + tilt_arr[ids_b][None, :]
        block = (
            joint_cost
            + (linf_base > reconfig_threshold_rad).astype(np.int64) * reconfig_unit_base
            + (linf_any > reconfig_threshold_rad).astype(np.int64) * reconfig_unit_any
            + edge_tilt * tilt_unit
        )
        costs[np.ix_(ids_a, ids_b)] = block
        costs[np.ix_(ids_b, ids_a)] = block.T

    # Dummy opens the cycle. Endpoint tilt is included so endpoints, like every
    # internal selected vertex, contribute their tilt cost exactly twice.
    costs[dummy_vertex, :dummy_vertex] = tilt_arr * tilt_unit
    costs[:dummy_vertex, dummy_vertex] = tilt_arr * tilt_unit
    costs[dummy_vertex, dummy_vertex] = forbidden_cost

    return {
        "sets": sets,
        "costs": costs,
        "vertex_viewpoint": np.asarray(vertex_viewpoint, dtype=np.int32),
        "vertex_candidate": np.asarray(vertex_candidate, dtype=np.int32),
        "dummy_vertex": int(dummy_vertex),
        "reconfig_unit": int(reconfig_unit_base),
        "reconfig_unit_base": int(reconfig_unit_base),
        "reconfig_unit_any": int(reconfig_unit_any),
        "reconfig_unit_wrist": int(reconfig_unit_any),  # v1 consumer compatibility
        "tilt_unit": int(tilt_unit),
        "max_l2_sum": int(max_l2_sum),
        "max_tilt_sum": int(max_tilt_sum),
        "max_any_sum": int(max_any_sum),
        "max_allowed_tour": int(max_allowed_tour),
        "forbidden_cost": int(forbidden_cost),
        "joint_cost_scale": int(joint_cost_scale),
        "joint_weights": weights,
        "reconfig_base_joints": base_idx,
        "reconfig_any_joints": any_idx,
        "allowed_view_edges": allowed_view_edges,
    }


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


def canonical_edge_set(edges: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges}

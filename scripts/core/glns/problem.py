"""Pure-Python GLNS graph, candidate, and GTSP problem construction."""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
from ortools.sat.python import cp_model
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# 주기성 규약은 core/trajectory/periodic.py 한 곳이 소유한다. 여기서는 쓰기만 한다
# (예전에는 이 파일이 갖고 있어서 trajectory -> glns 역방향 import 가 생겼다).
from core.trajectory.periodic import periodic_joint_delta, unwrap_joint_path
# 순수 그래프 연산이라 components_from_edges 옆(core/viewpoint/adjacency.py)이 소유한다.
# 여기서는 기존 import 경로 호환을 위해 재수출만 한다.
from core.viewpoint.adjacency import (  # noqa: F401
    canonical_edge_set,
    expand_edges_by_hops,
)

RESULT_FORMAT_VERSION = 2
JOINT_COST_SCALE = 1000
# 1계층(base 이동량)의 눈금. 이진 임계가 아니라 등급이라 크기가 비용에 반영된다.
# 거칠게 잡는 데는 이유가 둘 있다. (1) 곱게 잡으면 계층 단위가 성분 크기의 고차로 커져
# Int64 를 넘는다(1 mrad 은 성분 400개에서 터진다). (2) 같은 칸 안에서는 하위 계층
# (6축 reconfig → tilt → 가중 L2)이 계속 동점을 깨준다 — 눈금이 곱을수록 1계층이
# 단독 목적함수가 되어 tilt·wrist 선호가 죽는다.
BASE_TRAVEL_BUCKET_RAD = np.deg2rad(10.0)


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
    joint_periods: np.ndarray | None = None,
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

    # Compute every undirected incident-edge compatibility matrix once and
    # update both endpoint score arrays. The previous per-viewpoint loop built
    # the same A×B matrix again as B×A.
    base_connectivity = [np.zeros(len(reps), dtype=np.int32) for reps in representatives]
    any_connectivity = [np.zeros(len(reps), dtype=np.int32) for reps in representatives]
    needs_pruning = [
        len(representatives[i]) > max(1, int(cap_by_viewpoint[i])) for i in range(n)
    ]
    unique_edges = sorted({
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in np.asarray(edges, dtype=np.int32)
        if int(a) != int(b)
    })
    for a, b in unique_edges:
        if not needs_pruning[a] and not needs_pruning[b]:
            continue
        reps_a = np.asarray(representatives[a], dtype=np.float64)
        reps_b = np.asarray(representatives[b], dtype=np.float64)
        if not len(reps_a) or not len(reps_b):
            continue
        delta = np.abs(periodic_joint_delta(
            reps_a[:, None, :] - reps_b[None, :, :], joint_periods,
        ))
        base_ok = np.max(delta[..., :3], axis=2) <= threshold_rad
        any_ok = np.max(delta, axis=2) <= threshold_rad
        if needs_pruning[a]:
            base_connectivity[a] += np.any(base_ok, axis=1)
            any_connectivity[a] += np.any(any_ok, axis=1)
        if needs_pruning[b]:
            base_connectivity[b] += np.any(base_ok, axis=0)
            any_connectivity[b] += np.any(any_ok, axis=0)

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
            base_conn = base_connectivity[vp]
            any_conn = any_connectivity[vp]
            variants = np.asarray(md["variant"]).astype("U")
            tilt = np.asarray(md["tilt_deg"], dtype=np.float64)
            distance = np.linalg.norm(
                periodic_joint_delta(reps - ref, joint_periods) * weights, axis=1,
            )
            chosen: list[int] = []
            # Incrementally maintain each candidate's distance to the nearest
            # selected branch. Recomputing min(distance(candidate, every chosen))
            # inside the sort key caused millions of tiny NumPy calls.
            min_diversity = np.full(k, np.inf, dtype=np.float64)

            def choose_from(pool: list[int], limit: int) -> None:
                remaining = list(pool)
                while remaining and len(chosen) < limit:
                    def key(i: int):
                        return (-int(base_conn[i]), -int(any_conn[i]), float(tilt[i]),
                                -float(min_diversity[i]), float(distance[i]), int(i))
                    best = min(remaining, key=key)
                    chosen.append(best)
                    remaining.remove(best)
                    delta_to_best = periodic_joint_delta(
                        reps - reps[best], joint_periods,
                    )
                    np.minimum(
                        min_diversity,
                        np.linalg.norm(delta_to_best * weights, axis=1),
                        out=min_diversity,
                    )

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
    base_bucket_rad: float = BASE_TRAVEL_BUCKET_RAD,
    candidate_tilt_costs: list[np.ndarray] | None = None,
    joint_periods: np.ndarray | None = None,
) -> dict:
    """Build a complete integer GTSP matrix with Delaunay non-edges forbidden.

    Every source viewpoint is one GTSP set and each collision-free IK
    representative is a vertex in that set. A final singleton dummy set opens
    the otherwise cyclic GLNS tour.

    The integer objective is strict lexicographic, in this order: number of
    base-joint (q0:q3) reconfigurations, number of any-six-joint
    reconfigurations, quantised base-joint travel, accumulated vertex tilt cost,
    weighted joint L2 travel.

    Tier 1 grades base travel in ``base_bucket_rad`` steps instead of counting
    threshold crossings. Counting made a 170-deg base swing cost exactly as much
    as a 30-deg one, and made any number of sub-threshold moves free — measured
    on real solutions, 100% of base rotation sat under the threshold and was
    therefore invisible to the objective.
    Tilt vertex costs are placed on both incident symmetric edges, including
    dummy endpoint edges, so every selected pose contributes exactly twice.

    Tier 1/2 join indices are fixed at q0:q3 and q0:q6 to match the verifier —
    they are not configurable, because the verifier recomputes reconfiguration
    boundaries with the same fixed indices and the two must not drift apart.
    """
    members = np.asarray(members, dtype=np.int32)
    if len(members) < 2:
        raise ValueError("GTSP construction requires at least two viewpoints")
    if reconfig_threshold_rad <= 0.0:
        raise ValueError("reconfig_threshold_rad must be > 0")
    if joint_cost_scale <= 0:
        raise ValueError("joint_cost_scale must be > 0")
    if base_bucket_rad <= 0.0:
        raise ValueError("base_bucket_rad must be > 0")

    weights = (np.ones(6, dtype=np.float64) if joint_weights is None
               else np.asarray(joint_weights, dtype=np.float64))
    if weights.shape != (6,) or np.any(weights < 0.0):
        raise ValueError("joint_weights must be 6 non-negative values")
    periods = (np.zeros(6, dtype=np.float64) if joint_periods is None
               else np.asarray(joint_periods, dtype=np.float64))
    if periods.shape != (6,) or np.any(periods < 0.0):
        raise ValueError("joint_periods must be 6 non-negative values")
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
    max_base_bucket = 0
    for a, b in sorted(allowed_view_edges):
        qa = np.asarray(representatives[a], dtype=np.float64)
        qb = np.asarray(representatives[b], dtype=np.float64)
        diff = periodic_joint_delta(
            qa[:, None, :] - qb[None, :, :], periods,
        )
        # 2차(동점 깨기) 비용은 per-joint 가중 L2 — wrist roll 등 싼 관절은 작게 반영.
        l2 = np.linalg.norm(diff * weights[None, None, :], axis=2)
        joint_cost = np.rint(l2 * joint_cost_scale).astype(np.int64)
        if joint_cost.size:
            max_joint_cost = max(max_joint_cost, int(joint_cost.max()))
        absd = np.abs(diff)
        # Strict tiers use base q0:q3 and all joints q0:q6.
        # 1계층은 base 이동량을 눈금으로 끊어 '얼마나' 움직이는지를 센다(개수가 아니다).
        linf_base = np.max(absd[..., base_idx], axis=2)
        linf_any = np.max(absd[..., any_idx], axis=2)
        base_buckets = np.floor(linf_base / base_bucket_rad).astype(np.int64)
        if base_buckets.size:
            max_base_bucket = max(max_base_bucket, int(base_buckets.max()))
        blocks.append((ranges[a], ranges[b], joint_cost, linf_base, base_buckets, linf_any))

    # Exact tier bounds. Each unit is one larger than the maximum total cost of
    # all lower tiers, therefore no amount of lower-tier improvement can trade
    # against one unit in a higher tier.
    n_real_edges = len(members) - 1
    max_l2_sum = n_real_edges * max_joint_cost
    max_tilt_vertex = max(tilt_by_vertex, default=0)
    tilt_unit = max_l2_sum + 1
    max_tilt_sum = 2 * len(members) * max_tilt_vertex * tilt_unit
    # 계층: base reconfig 개수 → 6축 reconfig 개수 → base 이동량 버킷 → tilt → L2.
    #
    # 버킷은 두 reconfig 개수를 **대체하지 않고 그 아래에 더한다.** 실측으로 확인:
    #   버킷을 최상위로 교체 → scan base -22% 인데 reconfig 2->4, transit base +178%,
    #                          실행 총량 +45% 악화
    #   reconfig 개수만 위로   → base=0->1 로 팔 분기가 flip, transit base +226% 악화
    # reconfig 간선은 verify 에서 **계획된 transit** 을 부르고, 그 transit 이 사다리 끝에서
    # via-home 으로 빠지면 목적함수가 보지 못하는 큰 base 스윙이 된다. 특히 base reconfig
    # 하나는 팔 분기 flip 이라 그 자체가 거대한 이동이다. 그래서 개수를 먼저 0 으로 누르고,
    # **그 안에서** 임계 아래 누적을 버킷으로 줄인다.
    base_travel_unit = max_tilt_sum + max_l2_sum + 1
    max_base_sum = n_real_edges * max_base_bucket * base_travel_unit
    reconfig_unit_any = max_base_sum + max_tilt_sum + max_l2_sum + 1
    max_any_sum = n_real_edges * reconfig_unit_any
    reconfig_unit_base = max_any_sum + max_base_sum + max_tilt_sum + max_l2_sum + 1
    max_allowed_tour = (
        n_real_edges * reconfig_unit_base
        + max_any_sum + max_base_sum + max_tilt_sum + max_l2_sum
    )
    forbidden_cost = max_allowed_tour + 1
    if forbidden_cost > np.iinfo(np.int64).max:
        raise OverflowError(
            "strict lexicographic GTSP costs exceed Int64 "
            f"({len(members)} viewpoints, bucket={np.rad2deg(base_bucket_rad):.1f} deg). "
            "Use a coarser base_bucket_rad or split the component.")
    costs = np.full((n_vertices, n_vertices), forbidden_cost, dtype=np.int64)
    tilt_arr = np.asarray(tilt_by_vertex, dtype=np.int64)

    for ids_a, ids_b, joint_cost, linf_base, base_buckets, linf_any in blocks:
        edge_tilt = tilt_arr[ids_a, None] + tilt_arr[ids_b][None, :]
        block = (
            joint_cost
            + (linf_base > reconfig_threshold_rad).astype(np.int64) * reconfig_unit_base
            + (linf_any > reconfig_threshold_rad).astype(np.int64) * reconfig_unit_any
            + base_buckets * base_travel_unit
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
        "reconfig_unit_base": int(reconfig_unit_base),
        "base_travel_unit": int(base_travel_unit),
        "base_bucket_rad": float(base_bucket_rad),
        "max_base_bucket": int(max_base_bucket),
        "reconfig_unit_any": int(reconfig_unit_any),
        "tilt_unit": int(tilt_unit),
        "max_l2_sum": int(max_l2_sum),
        "max_tilt_sum": int(max_tilt_sum),
        "max_any_sum": int(max_any_sum),
        "max_base_sum": int(max_base_sum),
        "max_allowed_tour": int(max_allowed_tour),
        "forbidden_cost": int(forbidden_cost),
        "joint_cost_scale": int(joint_cost_scale),
        "joint_weights": weights,
        "joint_periods": periods,
        "reconfig_base_joints": base_idx,
        "reconfig_any_joints": any_idx,
        "allowed_view_edges": allowed_view_edges,
    }

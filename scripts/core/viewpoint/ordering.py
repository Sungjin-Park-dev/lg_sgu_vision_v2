"""Cluster ordering and path construction."""

from __future__ import annotations

import itertools

import numpy as np

from .adjacency import _tangent_basis
from .sampling import compute_pca_axes, reorder_zigzag

def _edge_cost(pos_from, pos_to, nrm_from, nrm_to, normal_weight):
    """두 점 사이의 위치+법선 비용 (미터 단위)."""
    d = float(np.linalg.norm(pos_from - pos_to))
    if normal_weight > 0:
        d += float(np.linalg.norm(nrm_from - nrm_to)) * normal_weight
    return d


def _gtsp_bruteforce(unique_clusters, cluster_internal, normal_weight):
    """K ≤ 2일 때 전수 탐색으로 최적 순서+방향을 반환."""
    import itertools
    K = len(unique_clusters)
    best_cost = np.inf
    best_order = None
    best_dirs = None

    for perm in itertools.permutations(range(K)):
        for dirs in itertools.product([0, 1], repeat=K):
            cost = 0.0
            for step in range(K - 1):
                k, l = perm[step], perm[step + 1]
                cid_k, cid_l = unique_clusters[k], unique_clusters[l]
                ci_k, ci_l = cluster_internal[cid_k], cluster_internal[cid_l]
                # 퇴장점: F=endpoint_b, R=endpoint_a
                exit_pos = ci_k['endpoint_a'] if dirs[step] == 1 else ci_k['endpoint_b']
                exit_nrm = ci_k['normal_a'] if dirs[step] == 1 else ci_k['normal_b']
                # 진입점: F=endpoint_a, R=endpoint_b
                entry_pos = ci_l['endpoint_b'] if dirs[step + 1] == 1 else ci_l['endpoint_a']
                entry_nrm = ci_l['normal_b'] if dirs[step + 1] == 1 else ci_l['normal_a']
                cost += _edge_cost(exit_pos, entry_pos, exit_nrm, entry_nrm, normal_weight)
            if cost < best_cost:
                best_cost = cost
                best_order = [unique_clusters[p] for p in perm]
                best_dirs = list(dirs)

    return (np.array(best_order, dtype=np.int32),
            np.array(best_dirs, dtype=np.int32))


def _gtsp_greedy_nn(unique_clusters, cluster_internal, normal_weight):
    """양방향 고려 greedy nearest-neighbor fallback."""
    K = len(unique_clusters)
    visited = set()
    order = []
    directions = []

    # 시작: 모든 클러스터×방향 중 가장 낮은 "첫 진입 비용"은 없으므로 임의로 첫 번째
    current_cid = unique_clusters[0]
    current_dir = 0
    order.append(current_cid)
    directions.append(current_dir)
    visited.add(current_cid)

    for _ in range(K - 1):
        ci_cur = cluster_internal[current_cid]
        cur_exit = ci_cur['endpoint_a'] if current_dir == 1 else ci_cur['endpoint_b']
        cur_exit_n = ci_cur['normal_a'] if current_dir == 1 else ci_cur['normal_b']

        best_cost = np.inf
        best_cid = None
        best_dir = 0
        for cid in unique_clusters:
            if cid in visited:
                continue
            ci = cluster_internal[cid]
            for d in [0, 1]:
                entry = ci['endpoint_b'] if d == 1 else ci['endpoint_a']
                entry_n = ci['normal_b'] if d == 1 else ci['normal_a']
                c = _edge_cost(cur_exit, entry, cur_exit_n, entry_n, normal_weight)
                if c < best_cost:
                    best_cost = c
                    best_cid = cid
                    best_dir = d

        order.append(best_cid)
        directions.append(best_dir)
        visited.add(best_cid)
        current_cid = best_cid
        current_dir = best_dir

    return (np.array(order, dtype=np.int32),
            np.array(directions, dtype=np.int32))


def order_clusters_gtsp(
    cluster_ids: np.ndarray,
    camera_positions: np.ndarray,
    cluster_internal: dict,
    normal_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """GTSP (Noon-Bean 변환 + 더미 노드)로 클러스터 방문 순서 및 방향 최적화.

    각 클러스터에 대해 정방향(F: a→b)과 역방향(R: b→a) 두 선택지를 두고,
    Noon-Bean 변환으로 GTSP를 ATSP로 변환하여 OR-Tools로 풀어
    최적 순서와 방향을 동시에 결정한다.
    더미 노드를 추가하여 open path(비순환 경로)를 생성한다.

    Args:
        cluster_ids: (N,) 0-based 클러스터 할당
        camera_positions: (N, 3)
        cluster_internal: compute_cluster_internal_order()의 결과
        normal_weight: 법선 가중치 (0이면 위치만 사용)

    Returns:
        cluster_order: (K,) 클러스터 방문 순서 (클러스터 ID 배열)
        cluster_direction: (K,) 각 클러스터의 방향 (0=Forward a→b, 1=Reverse b→a)
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    unique_clusters = np.unique(cluster_ids)
    K = len(unique_clusters)

    if K <= 2:
        return _gtsp_bruteforce(unique_clusters, cluster_internal, normal_weight)

    # --- 엔드포인트 배열 준비 ---
    ep_a = np.array([cluster_internal[c]['endpoint_a'] for c in unique_clusters])
    ep_b = np.array([cluster_internal[c]['endpoint_b'] for c in unique_clusters])
    nr_a = np.array([cluster_internal[c]['normal_a'] for c in unique_clusters])
    nr_b = np.array([cluster_internal[c]['normal_b'] for c in unique_clusters])

    # --- Noon-Bean ATSP 거리행렬 구성 (2K+1 노드) ---
    # 노드 인덱싱: 2*k = F_k, 2*k+1 = R_k, 2*K = Dummy
    SCALE = 1_000_000
    INF = 10**15
    N_ATSP = 2 * K + 1
    D = 2 * K  # dummy node index

    atsp = np.full((N_ATSP, N_ATSP), INF, dtype=np.int64)

    for k in range(K):
        # 클러스터 내 사이클 (비용 0)
        atsp[2 * k, 2 * k + 1] = 0      # F_k → R_k
        atsp[2 * k + 1, 2 * k] = 0      # R_k → F_k

        for l in range(K):
            if k == l:
                continue

            # 원본 GTSP 비용 (퇴장점 → 진입점)
            # F_k→F_l: exit_b[k] → entry_a[l]
            ff = _edge_cost(ep_b[k], ep_a[l], nr_b[k], nr_a[l], normal_weight)
            # F_k→R_l: exit_b[k] → entry_b[l]
            fr = _edge_cost(ep_b[k], ep_b[l], nr_b[k], nr_b[l], normal_weight)
            # R_k→F_l: exit_a[k] → entry_a[l]
            rf = _edge_cost(ep_a[k], ep_a[l], nr_a[k], nr_a[l], normal_weight)
            # R_k→R_l: exit_a[k] → entry_b[l]
            rr = _edge_cost(ep_a[k], ep_b[l], nr_a[k], nr_b[l], normal_weight)

            # Noon-Bean: GTSP[X_k, Y_l] → ATSP[pred(X_k), Y_l]
            # pred(F_k) = R_k, pred(R_k) = F_k
            atsp[2 * k + 1, 2 * l]     = int(ff * SCALE)  # GTSP F_k→F_l → ATSP R_k→F_l
            atsp[2 * k + 1, 2 * l + 1] = int(fr * SCALE)  # GTSP F_k→R_l → ATSP R_k→R_l
            atsp[2 * k,     2 * l]     = int(rf * SCALE)   # GTSP R_k→F_l → ATSP F_k→F_l
            atsp[2 * k,     2 * l + 1] = int(rr * SCALE)   # GTSP R_k→R_l → ATSP F_k→R_l

    # 더미 노드: open path를 위해 비용 0
    atsp[D, :] = 0
    atsp[:, D] = 0
    atsp[D, D] = INF

    # --- OR-Tools ATSP ---
    manager = pywrapcp.RoutingIndexManager(N_ATSP, 1, D)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return atsp[from_node, to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 2

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        # D를 제외한 투어 추출
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != D:
                tour.append(node)
            index = solution.Value(routing.NextVar(index))

        # 디코딩: 2개씩 쌍, 첫 번째가 진입 노드
        cluster_order = []
        cluster_direction = []
        for i in range(0, len(tour), 2):
            entry_node = tour[i]
            k = entry_node // 2
            direction = entry_node % 2  # 0=F, 1=R
            cluster_order.append(unique_clusters[k])
            cluster_direction.append(direction)

        return (np.array(cluster_order, dtype=np.int32),
                np.array(cluster_direction, dtype=np.int32))

    # Fallback: 양방향 greedy NN
    print("  Warning: GTSP solver failed, falling back to greedy NN")
    return _gtsp_greedy_nn(unique_clusters, cluster_internal, normal_weight)


def _two_opt_open(order: list, points: np.ndarray, max_passes: int) -> list:
    """Open-path 2-opt 개선(full). 양 끝점(order[0], order[-1])은 고정 → 열린 경로 유지.

    클러스터가 작으므로(호출측에서 n<=max_2opt_n 보장) O(n²) full 2-opt로 충분.
    """
    n = len(order)
    if n < 4:
        return order
    pos = points
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                a, b = order[i - 1], order[i]
                c, d = order[j], order[j + 1]
                before = np.linalg.norm(pos[a] - pos[b]) + np.linalg.norm(pos[c] - pos[d])
                after = np.linalg.norm(pos[a] - pos[c]) + np.linalg.norm(pos[b] - pos[d])
                if after + 1e-12 < before:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    return order


def order_cluster_graph(
    camera_positions_sub: np.ndarray,
    normals_sub: np.ndarray,
    max_2opt_n: int = 120,
    max_2opt_passes: int = 30,
) -> list:
    """한 클러스터의 nearest-neighbor open-path 순서(로컬 인덱스 permutation).

    전역 PCA 평면 대신 클러스터 평균 법선의 탄젠트 평면을 사용해 시작 극단점과
    tangent-정렬 baseline을 잡고, 카메라 위치 기준 nearest-neighbor + 2-opt로
    경로를 만든다. 절대 tangent-정렬 baseline보다 길어지지 않도록 가드한다.

    Returns: list[int] — 방문 순서(permutation). sorted_indices = [idx[i] for i in perm].
    """
    P = np.asarray(camera_positions_sub, dtype=np.float64)
    n = len(P)
    if n <= 2:
        return list(range(n))
    if not np.all(np.isfinite(P)):
        return list(range(n))

    # 1. 평균 법선 탄젠트 프레임 → 2D 투영
    mean_n = np.asarray(normals_sub, dtype=np.float64).mean(axis=0)
    u, v = _tangent_basis(mean_n)
    Pc = P - P.mean(axis=0)
    P2 = np.column_stack([Pc @ u, Pc @ v])

    # 최장 탄젠트 extent 축 → 시작점(극단) + tangent-정렬 baseline
    spread = P2.max(axis=0) - P2.min(axis=0)
    major = 0 if spread[0] >= spread[1] else 1
    tangent_sorted = list(np.argsort(P2[:, major]))

    # 2. nearest-neighbor seed (극단점에서 시작) + 2-opt
    start = int(np.argmin(P2[:, major]))
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        d = np.linalg.norm(P - P[cur], axis=1)
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt

    if n <= max_2opt_n:
        order = _two_opt_open(order, P, max_2opt_passes)

    # 3. anti-explosion 가드: tangent-정렬 baseline보다 길면 폴백
    def _plen(seq: list) -> float:
        return float(np.sum(np.linalg.norm(np.diff(P[seq], axis=0), axis=1))) if len(seq) > 1 else 0.0

    if _plen(order) > _plen(tangent_sorted):
        return tangent_sorted
    return order


def order_cluster_lawnmower(
    surface_positions_sub: np.ndarray,
    camera_positions_sub: np.ndarray,
    normals_sub: np.ndarray,
    row_spacing_m: float,
) -> list:
    """한 클러스터를 tangent-plane lawnmower 패턴으로 정렬한다.

    row/scan 축은 표면점 기준으로 잡고, 두 가능한 시작 방향 중 실제 카메라
    위치 경로가 짧은 쪽을 선택한다. 따라서 coverage row는 surface 기준,
    이동 비용은 working-distance가 반영된 camera 기준이 된다.
    """
    S = np.asarray(surface_positions_sub, dtype=np.float64)
    C = np.asarray(camera_positions_sub, dtype=np.float64)
    n = len(S)
    if n <= 2:
        return list(range(n))
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(C)):
        return order_cluster_graph(camera_positions_sub, normals_sub)

    spacing = max(float(row_spacing_m), 1e-6)

    # 1. 평균 법선 tangent frame으로 표면점을 2D 투영
    mean_n = np.asarray(normals_sub, dtype=np.float64).mean(axis=0)
    u, v = _tangent_basis(mean_n)
    Sc = S - S.mean(axis=0)
    P2 = np.column_stack([Sc @ u, Sc @ v])
    if not np.all(np.isfinite(P2)):
        return order_cluster_graph(camera_positions_sub, normals_sub)

    # 2. tangent 2D 안에서 PCA: 긴 축=scan, 짧은 축=row
    P2c = P2 - P2.mean(axis=0)
    try:
        cov = np.cov(P2c, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        scan_axis = vecs[:, order[0]]
        row_axis = vecs[:, order[1]]
    except Exception:  # noqa: BLE001
        spread = P2.max(axis=0) - P2.min(axis=0)
        scan_axis = np.array([1.0, 0.0]) if spread[0] >= spread[1] else np.array([0.0, 1.0])
        row_axis = np.array([-scan_axis[1], scan_axis[0]])

    scan = P2c @ scan_axis
    row = P2c @ row_axis

    # 3. FOV-derived spacing으로 row binning
    row_span = float(row.max() - row.min())
    if row_span < spacing * 0.5:
        row_bins = np.zeros(n, dtype=np.int32)
    else:
        row_bins = np.floor((row - row.min()) / spacing + 0.5).astype(np.int32)

    rows = []
    for rb in np.unique(row_bins):
        idx = np.where(row_bins == rb)[0]
        if idx.size == 0:
            continue
        rows.append((float(row[idx].mean()), idx))
    rows.sort(key=lambda item: item[0])

    if not rows:
        return order_cluster_graph(camera_positions_sub, normals_sub)

    def _make(reverse_first: bool) -> list:
        out = []
        for r, (_, idx) in enumerate(rows):
            local = idx[np.argsort(scan[idx], kind="stable")]
            if (r % 2 == 1) ^ reverse_first:
                local = local[::-1]
            out.extend(int(i) for i in local)
        return out

    def _plen(seq: list) -> float:
        return float(np.sum(np.linalg.norm(np.diff(C[seq], axis=0), axis=1))) if len(seq) > 1 else 0.0

    forward = _make(False)
    reverse = _make(True)
    return reverse if _plen(reverse) < _plen(forward) else forward


def compute_cluster_internal_order(
    cluster_ids: np.ndarray,
    surface_positions: np.ndarray,
    camera_positions: np.ndarray,
    normals: np.ndarray,
    row_spacing_m: float,
    col_spacing_m: Optional[float] = None,
    grid_row_index: Optional[np.ndarray] = None,
    global_axis1: Optional[np.ndarray] = None,
    global_axis2: Optional[np.ndarray] = None,
    ordering_mode: str = 'zigzag',
) -> dict:
    """각 클러스터의 내부 방문 순서를 사전 계산.

    TSP 전에 호출하여 클러스터별 start/end point를 확보한다.

    grid_row_index가 주어지면 전역 축(global_axis1/axis2)을 사용하여
    클러스터별 PCA를 생략한다. row_index_override가 행 구분을 담당하므로
    로컬 PCA의 axis1은 불필요하고, axis2는 클러스터 간 열 방향 일관성을
    위해 이미 전역 값을 사용해야 하기 때문이다.

    Args:
        cluster_ids: (N,) 클러스터 ID
        surface_positions: (N, 3) 표면 위치
        camera_positions: (N, 3) 카메라 위치
        normals: (N, 3) 법선
        row_spacing_m: 행 간격 (미터)
        col_spacing_m: 열 간격 (미터). lawnmower row 간격은 min(row, col)을 사용.
        grid_row_index: (N,) 그리드 생성 시 할당된 원본 행 인덱스. None이면 양자화로 추정.
        global_axis1: (3,) 전역 행 방향 축. grid_row_index 사용 시 전달 (proj1 계산용, 행 구분에는 미사용).
        global_axis2: (3,) 전역 열 방향 축. grid_row_index 사용 시 행 내 정렬에 사용.

    Returns:
        dict[cluster_id] → {
            'sorted_indices': list[int],  # 내부 순서대로 정렬된 원본 인덱스
            'endpoint_a': (3,),           # 카메라 끝점 A
            'endpoint_b': (3,),           # 카메라 끝점 B
            'normal_a': (3,),             # 끝점 A 법선
            'normal_b': (3,),             # 끝점 B 법선
        }
    """
    unique_clusters = np.unique(cluster_ids)
    result = {}

    for cid in unique_clusters:
        mask = cluster_ids == cid
        indices = np.where(mask)[0]

        if len(indices) < 3:
            sorted_indices = list(indices)
        elif ordering_mode == 'lawnmower':
            spacing = min(row_spacing_m, col_spacing_m) if col_spacing_m is not None else row_spacing_m
            perm = order_cluster_lawnmower(
                surface_positions[indices], camera_positions[indices], normals[indices], spacing,
            )
            sorted_indices = [indices[i] for i in perm]
        elif ordering_mode == 'graph':
            # 평균 법선 tangent 기준 시작점 + camera-space NN + open-path 2-opt.
            # order_cluster_graph는 permutation을 직접 반환(argsort 불필요).
            perm = order_cluster_graph(camera_positions[indices], normals[indices])
            sorted_indices = [indices[i] for i in perm]
        else:
            cluster_cam = camera_positions[indices]

            if grid_row_index is not None:
                cluster_row_idx = grid_row_index[indices]
                axis1 = global_axis1 / np.linalg.norm(global_axis1)
                axis2 = global_axis2 / np.linalg.norm(global_axis2)
                local_order, _, _ = reorder_zigzag(
                    cluster_cam, axis1, axis2, row_spacing_m,
                    row_index_override=cluster_row_idx,
                )
            else:
                _, axis1, axis2 = compute_pca_axes(cluster_cam.astype(np.float64))
                local_order, _, _ = reorder_zigzag(cluster_cam, axis1, axis2, row_spacing_m)

            sorted_local = np.argsort(local_order)
            sorted_indices = [indices[i] for i in sorted_local]

        result[cid] = {
            'sorted_indices': sorted_indices,
            'endpoint_a': camera_positions[sorted_indices[0]],
            'endpoint_b': camera_positions[sorted_indices[-1]],
            'normal_a': normals[sorted_indices[0]],
            'normal_b': normals[sorted_indices[-1]],
        }

    return result


def build_clustered_path_order(
    cluster_ids: np.ndarray,
    cluster_order: np.ndarray,
    cluster_internal: dict,
    cluster_direction: Optional[np.ndarray] = None,
) -> np.ndarray:
    """클러스터 순서와 사전 계산된 내부 순서를 결합하여 글로벌 path_order 생성.

    Args:
        cluster_ids: (N,) 클러스터 할당
        cluster_order: (K,) 클러스터 방문 순서
        cluster_internal: compute_cluster_internal_order()의 결과
        cluster_direction: (K,) 각 클러스터의 방향 (0=Forward, 1=Reverse). None이면 전부 Forward.

    Returns:
        path_order: (N,) 글로벌 경로 순서
    """
    N = len(cluster_ids)
    path_order = np.zeros(N, dtype=np.int32)
    global_idx = 0

    for rank, cid in enumerate(cluster_order):
        indices = cluster_internal[cid]['sorted_indices']
        if cluster_direction is not None and cluster_direction[rank] == 1:
            indices = list(reversed(indices))
        for idx in indices:
            path_order[idx] = global_idx
            global_idx += 1

    return path_order


# ============================================================================
# HDF5 I/O
# ============================================================================

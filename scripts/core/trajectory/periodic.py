"""관절 주기성(2π) 규약 한 곳.

UR20 은 여섯 축 모두 ±354° 라 **같은 자세를 여러 관절값으로 표현할 수 있다.** 그 사실을
다루는 방식이 예전에는 세 파일에 흩어져 있었고(ik.py 의 normalize, glns/problem.py 의
delta·unwrap), 성분을 잇는 joining.py 는 아무것도 쓰지 않아 같은 자세를 302° 떨어진 것으로
보고 그만큼 실제로 돌았다. 규약을 여기 모아 그 종류의 버그가 한 곳에서만 관리되게 한다.

네 가지가 각각 다른 일을 한다 — 이름만 보고 섞어 쓰면 안 된다:

    normalize_joints(q)          값을 [-π, π] 로 접는다. IK 직후 표준화용.
    periodic_joint_delta(d)      **차이**를 접는다. 값은 그대로 — 비용 계산용.
    unwrap_joint_path(path)      경로 전체를 한계 안에서 최적으로 편다(연속화).
    align_to_reference(q, ref)   목표를 기준에 가장 가까운 등가로 옮긴다. 이동 계획 직전.

trajectory 밑에 두는 이유: glns 가 trajectory 를 import 하는 정방향은 자연스럽지만
반대는 순환이다. 주기성은 가장 밑바닥 규약이라 여기 있어야 양쪽이 같이 볼 수 있다.
"""

from __future__ import annotations

import itertools

import numpy as np


def normalize_joints(q):
    """Joint angles를 [-π, π] 범위로 정규화. 형상 유지."""
    return ((q + np.pi) % (2 * np.pi)) - np.pi


def periodic_joint_delta(delta: np.ndarray, joint_periods: np.ndarray | None = None) -> np.ndarray:
    """Return signed shortest deltas for periodic joints, preserving array shape."""
    out = np.asarray(delta, dtype=np.float64).copy()
    if joint_periods is None:
        return out
    periods = np.asarray(joint_periods, dtype=np.float64)
    if out.shape[-1] != len(periods) or np.any(periods < 0.0):
        raise ValueError("joint_periods must be non-negative and match the final dimension")
    mask = periods > 0.0
    out[..., mask] = (
        (out[..., mask] + periods[mask] / 2.0) % periods[mask]
        - periods[mask] / 2.0
    )
    return out


def unwrap_joint_path(
    path: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    joint_periods: np.ndarray,
    threshold_rad: float,
    joint_weights: np.ndarray | None = None,
    reference_joints: np.ndarray | None = None,
) -> np.ndarray:
    """Choose limit-valid 2π-equivalent configurations for an entire open path.

    Dynamic programming preserves strict base-reconfiguration → any-joint
    reconfiguration → weighted-L2 ordering. Endpoint L2 distance to the optional
    reference breaks globally shifted ties without changing reconfiguration tiers.
    """
    q_path = np.asarray(path, dtype=np.float64)
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    periods = np.asarray(joint_periods, dtype=np.float64)
    weights = (np.ones(q_path.shape[1], dtype=np.float64) if joint_weights is None
               else np.asarray(joint_weights, dtype=np.float64))
    reference = (None if reference_joints is None
                 else np.asarray(reference_joints, dtype=np.float64))
    if q_path.ndim != 2 or q_path.shape[1] == 0:
        raise ValueError("path must have shape (N, dof)")
    dof = q_path.shape[1]
    if any(x.shape != (dof,) for x in (lower, upper, periods, weights)):
        raise ValueError("joint limit/period/weight arrays must match path dof")
    if reference is not None and reference.shape != (dof,):
        raise ValueError("reference_joints must match path dof")
    if np.any(lower > upper) or threshold_rad <= 0.0:
        raise ValueError("invalid joint limits or threshold")
    if len(q_path) == 0:
        return q_path.copy()

    states: list[np.ndarray] = []
    tol = 1e-9
    for q in q_path:
        choices = []
        for j, value in enumerate(q):
            if periods[j] > 0.0:
                k_min = int(np.ceil((lower[j] - value - tol) / periods[j]))
                k_max = int(np.floor((upper[j] - value + tol) / periods[j]))
                vals = [value + k * periods[j] for k in range(k_min, k_max + 1)]
            else:
                vals = [float(value)] if lower[j] - tol <= value <= upper[j] + tol else []
            if not vals:
                raise ValueError(f"joint {j} has no equivalent value inside its limits")
            choices.append(vals)
        states.append(np.asarray(list(itertools.product(*choices)), dtype=np.float64))

    # Each state cost is a strict tuple: (base count, any count, weighted L2).
    prev_cost = [
        (0, 0, 0.0 if reference is None
         else float(np.linalg.norm((s - reference) * weights)))
        for s in states[0]
    ]
    predecessors: list[np.ndarray] = []
    for step in range(1, len(states)):
        prev_states, cur_states = states[step - 1], states[step]
        cur_cost = []
        cur_pred = np.empty(len(cur_states), dtype=np.int32)
        for ci, cur in enumerate(cur_states):
            best_cost, best_pi = None, -1
            for pi, prev in enumerate(prev_states):
                delta = np.abs(cur - prev)
                edge = (
                    int(np.max(delta[:3]) > threshold_rad),
                    int(np.max(delta) > threshold_rad),
                    float(np.linalg.norm(delta * weights)),
                )
                candidate = (
                    prev_cost[pi][0] + edge[0],
                    prev_cost[pi][1] + edge[1],
                    prev_cost[pi][2] + edge[2],
                )
                if best_cost is None or candidate < best_cost:
                    best_cost, best_pi = candidate, pi
            cur_cost.append(best_cost)
            cur_pred[ci] = best_pi
        predecessors.append(cur_pred)
        prev_cost = cur_cost

    if reference is None:
        final = min(range(len(prev_cost)), key=lambda i: (prev_cost[i], i))
    else:
        final = min(
            range(len(prev_cost)),
            key=lambda i: (
                prev_cost[i][0], prev_cost[i][1],
                prev_cost[i][2] + float(np.linalg.norm((states[-1][i] - reference) * weights)),
                i,
            ),
        )
    indices = [final]
    for pred in reversed(predecessors):
        indices.append(int(pred[indices[-1]]))
    indices.reverse()
    return np.stack([states[i][indices[i]] for i in range(len(states))])


def align_to_reference(
    target: np.ndarray,
    reference: np.ndarray,
    joint_periods: np.ndarray,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
) -> np.ndarray:
    """``target`` 을 ``reference`` 에 가장 가까운 2π 등가로 옮긴다 (관절 한계 안에서).

    같은 자세를 가리키는 여러 관절값 중 **덜 움직이는 것**을 고른다. 두 자세 사이를
    계획하기 직전에 쓴다 — 안 쓰면 154° 에서 -148° 로 가라는 요청이 302° 회전이 된다
    (같은 자세인데 211.9° 로 가면 58° 다).

    한계를 주면 그 안에 있는 후보만 고른다. 후보가 하나도 없으면 원래 값을 그대로
    돌려준다 — 조용히 한계를 넘는 값을 만들지 않는다.
    """
    tgt = np.asarray(target, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    periods = np.asarray(joint_periods, dtype=np.float64)
    if tgt.shape != ref.shape or tgt.shape[-1] != len(periods):
        raise ValueError("target/reference must share shape and match joint_periods")
    if np.any(periods < 0.0):
        raise ValueError("joint_periods must be non-negative")

    out = tgt.copy()
    lower = None if joint_lower is None else np.asarray(joint_lower, dtype=np.float64)
    upper = None if joint_upper is None else np.asarray(joint_upper, dtype=np.float64)
    tol = 1e-9
    for j, period in enumerate(periods):
        if period <= 0.0:
            continue
        # 접힌 차이를 기준에 더하면 그것이 곧 '가장 가까운 등가'다.
        best = ref[..., j] + periodic_joint_delta(
            (tgt[..., j] - ref[..., j])[..., None], np.array([period]))[..., 0]
        if lower is not None and upper is not None:
            # 한계 밖이면 한계 안에 들어오는 등가 중 기준에 가장 가까운 것으로 되돌린다.
            k = np.rint((best - tgt[..., j]) / period)
            for shift in (0.0, -1.0, 1.0, -2.0, 2.0):
                cand = tgt[..., j] + (k + shift) * period
                if np.all(cand >= lower[j] - tol) and np.all(cand <= upper[j] + tol):
                    best = cand
                    break
            else:
                best = tgt[..., j]      # 한계 안 후보 없음 — 건드리지 않는다
        out[..., j] = best
    return out

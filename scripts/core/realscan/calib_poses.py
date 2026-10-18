#!/usr/bin/env python3
"""손목 카메라 보정 자세를 규칙적으로 만든다 — 교시 대신 pose_tour 로 보낼 관절값 목록.

보드 중심 둘레의 카메라 시점 격자(거리 × 기울기 × 방위 × 광축 roll)를 만들고, 각 시점에 대해
PiPER IK 를 풀어 (1) 관절 한계 안 (2) 보드 전체가 RGB 영상 안에 여유 있게 (3) 칸이 충분히 크게
(4) 테이블 위로 충분히 높게 (5) 기존 교시 보정 자세와 비슷한 팔 모양 (6) 팔을 너무 멀리 뻗지 않는
것만 남긴다. 그중 손목 회전이 서로 가장 다른 것을 골라
학습·별도 검증 자세로 나누고, 이동이 짧은 순서로 정렬해 pose_tour --poses YAML 로 쓴다.

계산은 현재 보정(tool_T_color)과 보정 때 잰 보드 위치(base_T_board)를 쓴다 — 보드를 그 자리에
다시 놓는다는 가정이다. 장착이 크게 바뀌었으면 예측이 어긋나므로 먼저 몇 자세로 확인한다.
--evaluate 는 같은 잡음 모델로 이 자세 집합과 기존 교시 보정 자세의 hand-eye 오차를 비교한다.

    python scripts/core/realscan/calib_poses.py --out output/piper_calibration/generated --evaluate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from core.realscan.capture import rigid, transform  # noqa: E402
from core.realscan.piper import JOINT_NAMES, PiperFK  # noqa: E402
from core.reconstruction.sensor import look_at  # noqa: E402

BOARD_M = (0.12, 0.10)                         # 6×5칸 × 20mm — 보드 좌표 원점은 바깥 모서리
INNER = np.array([[x * 0.02, y * 0.02, 0.0] for y in range(1, 5) for x in range(1, 6)])  # 안쪽 코너 20개
OUTER = np.array([[0, 0, 0], [0.12, 0, 0], [0.12, 0.10, 0], [0, 0.10, 0]], dtype=float)
# 관절 한계 여유 — 손목(4·6)은 케이블 때문에 더 둔다
LIMIT_MARGIN_DEG = np.array([3, 3, 3, 8, 3, 8], dtype=float)


class FastFK:
    """URDF 관절 원점·축으로 직접 곱하는 FK — IK 반복용. PiperFK 와 같은 값인지 생성 시 확인한다."""

    def __init__(self, fk: PiperFK):
        chain = [fk.robot.joint_map[n] for n in JOINT_NAMES]
        self.origins = [np.asarray(j.origin, dtype=float) for j in chain]
        self.axes = [np.asarray(j.axis, dtype=float) / np.linalg.norm(j.axis) for j in chain]
        self.lower = np.array([j.limit.lower for j in chain])
        self.upper = np.array([j.limit.upper for j in chain])
        # base_link → joint1 원점 사이 고정 관절이 없는지(=체인이 base_link 에서 바로 시작하는지)
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.uniform(self.lower, self.upper)
            if not np.allclose(self.pose(q), fk.pose(q), atol=1e-9):
                raise RuntimeError("FastFK 가 URDF FK 와 다릅니다 — 체인 구조를 확인하세요")

    def frames(self, q):
        T = np.eye(4)
        out = []
        for origin, axis, v in zip(self.origins, self.axes, q):
            T = T @ origin @ rigid(Rotation.from_rotvec(axis * v).as_matrix(), np.zeros(3))
            out.append(T)
        return out

    def pose(self, q):
        return self.frames(q)[-1]


def solve_ik(ffk: FastFK, target, seeds, lo, hi):
    """link6 목표 자세에 가장 가까운 해 — 위치 0.3mm·회전 0.1° 안이면 성공. 여러 시드 중 시드에 가까운 해."""
    def residual(q):
        T = ffk.pose(q)
        rot = Rotation.from_matrix(target[:3, :3].T @ T[:3, :3]).as_rotvec()
        return np.concatenate([T[:3, 3] - target[:3, 3], 0.1 * rot])

    best = None
    for seed in seeds:
        x0 = np.clip(seed, lo + 1e-6, hi - 1e-6)
        r = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-10, ftol=1e-12, max_nfev=200)
        pos, rot = np.linalg.norm(r.fun[:3]), np.degrees(np.linalg.norm(r.fun[3:]) / 0.1)
        if pos < 3e-4 and rot < 0.1:
            move = float(np.max(np.abs(r.x - seed)))
            if best is None or move < best[1]:
                best = (r.x, move)
    return None if best is None else best[0]


def wrist_reach(frames):
    """joint2 축에서 손목(link6)까지 수평 거리 — 팔을 멀리 뻗을수록 j2·j3 중력 부하가 커진다.
    2026-10-18 실기: 33.6cm 자세에서 모터 소리가 달라지고 정지 판정이 깜빡였다(촬영은 됨)."""
    return float(np.linalg.norm(frames[5][:2, 3] - frames[1][:2, 3]))


def board_view(color_T_board, K, size, margin_px):
    """보드가 영상 안에 다 들어오는지와 칸 크기[px], 기울기[deg]."""
    W, H = size
    R, t = color_T_board[:3, :3], color_T_board[:3, 3]
    P = (R @ OUTER.T).T + t
    if np.any(P[:, 2] <= 0.05):
        return None
    uv = (K @ P.T).T
    uv = uv[:, :2] / uv[:, 2:]
    if uv[:, 0].min() < margin_px or uv[:, 1].min() < margin_px or \
            uv[:, 0].max() > W - margin_px or uv[:, 1].max() > H - margin_px:
        return None
    Pi = (R @ INNER.T).T + t
    ui = (K @ Pi.T).T
    ui = ui[:, :2] / ui[:, 2:]
    square = float(np.min(np.linalg.norm(np.diff(ui.reshape(4, 5, 2), axis=1), axis=2)))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
    return {"bbox_px": [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())],
            "min_square_px": square, "tilt_deg": tilt,
            "distance_m": float(np.linalg.norm(R @ [0.06, 0.05, 0] + t))}


def candidates(centre, distances, tilts, azimuths, rolls):
    for d in distances:
        for tilt in tilts:
            for az in (azimuths if tilt > 0 else [0.0]):
                a, e = np.radians(az), np.radians(90 - tilt)
                pos = centre + d * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
                for roll in rolls:
                    yield {"distance_m": d, "view_tilt_deg": tilt, "azimuth_deg": az, "roll_deg": roll,
                           "base_T_color": look_at(pos, centre, roll_deg=roll)}


def rot_deg(A, B):
    return float(np.degrees(Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude()))


def pick_diverse(pool, count, chosen=()):
    """손목(link6) 자세의 회전 차이 기준 maximin — 이미 고른 것과 가장 다른 것을 차례로 더한다."""
    chosen = list(chosen)
    picked = []
    rest = list(pool)
    if not chosen and rest:                    # 시작: 가장 덜 기운 정면 시점
        first = min(rest, key=lambda c: (c["view"]["tilt_deg"], c["distance_m"]))
        picked.append(first)
        rest.remove(first)
    while len(picked) < count and rest:
        ref = chosen + picked
        nxt = max(rest, key=lambda c: min(rot_deg(c["tool"], r["tool"]) for r in ref))
        picked.append(nxt)
        rest.remove(nxt)
    return picked


def pick_holdout(pool, count, train, novelty_deg):
    """별도 검증 자세 — 학습 자세 어느 것과도 lo° 이상 다르되 hi° 를 넘지 않는 것(학습 범위 안쪽의 새 시점).
    가장 튀는 자세를 고르면 팔 모델 오차(수 mm)까지 검증에 섞여 5mm 기준을 넘기 쉽다. 그 안에서 서로 maximin."""
    lo, hi = novelty_deg
    near = lambda c: min(rot_deg(c["tool"], t["tool"]) for t in train)
    window = [c for c in pool if lo <= near(c) <= hi]
    if not window:
        return []
    picked = [min(window, key=lambda c: (c["view"]["tilt_deg"], c["distance_m"]))]
    rest = [c for c in window if c is not picked[0]]
    while len(picked) < count and rest:
        nxt = max(rest, key=lambda c: min(rot_deg(c["tool"], r["tool"]) for r in picked))
        picked.append(nxt)
        rest.remove(nxt)
    return picked


def order_by_motion(items, start_q):
    """관절 L∞ 이동이 짧은 순서(최근접 이웃)."""
    rest, out, q = list(items), [], np.asarray(start_q)
    while rest:
        nxt = min(rest, key=lambda c: np.max(np.abs(c["q"] - q)))
        out.append(nxt)
        rest.remove(nxt)
        q = nxt["q"]
    return out


def hand_eye_error(ffk, qs, X, B, trials, seed=0, sigma_t_mm=(0.3, 0.3, 1.0), sigma_r_deg=0.15, sigma_q_deg=0.03):
    """잡음 넣은 가상 측정으로 cv2 PARK hand-eye 를 풀어 X 오차 분포 — 자세 집합끼리 비교용."""
    import cv2
    rng = np.random.default_rng(seed)
    tools = [ffk.pose(q) for q in qs]
    boards = [np.linalg.inv(T @ X) @ B for T in tools]
    errs = []
    for _ in range(trials):
        Ts, Cs = [], []
        for q, C in zip(qs, boards):
            Ts.append(ffk.pose(q + np.radians(rng.normal(0, sigma_q_deg, 6))))
            noise = rigid(Rotation.from_rotvec(np.radians(rng.normal(0, sigma_r_deg, 3))).as_matrix(),
                          rng.normal(0, 1, 3) * np.asarray(sigma_t_mm) / 1000)
            Cs.append(noise @ C)
        R, t = cv2.calibrateHandEye([T[:3, :3] for T in Ts], [T[:3, 3] for T in Ts],
                                    [C[:3, :3] for C in Cs], [C[:3, 3] for C in Cs], method=cv2.CALIB_HAND_EYE_PARK)
        E = np.linalg.inv(X) @ rigid(R, t.ravel())
        errs.append((np.linalg.norm(E[:3, 3]) * 1000, np.degrees(Rotation.from_matrix(E[:3, :3]).magnitude())))
    e = np.array(errs)
    return {"translation_mm_median": float(np.median(e[:, 0])), "translation_mm_p95": float(np.percentile(e[:, 0], 95)),
            "rotation_deg_median": float(np.median(e[:, 1])), "rotation_deg_p95": float(np.percentile(e[:, 1], 95))}


def write_yaml(path, items, note):
    import yaml
    lines = [f"# {note}", "# 각 줄: 관절값[deg] — 주석: 보드 거리·기울기·광축 roll·영상 속 보드 위치·최소 칸 크기"]
    lines.append("poses_deg:")
    for c in items:
        if "line" in c:                        # --keep-train 으로 유지한 줄은 그대로 (주석 포함)
            lines.append(c["line"])
            continue
        v = c["view"]
        lines.append(f"- {yaml.safe_dump([round(float(x), 3) for x in np.degrees(c['q'])], default_flow_style=True).strip()}"
                     f"  # {v['distance_m'] * 100:.0f}cm 기울기 {v['tilt_deg']:.0f}° roll {c['roll_deg']:+.0f}° "
                     f"bbox {[round(b) for b in v['bbox_px']]} 칸 {v['min_square_px']:.0f}px")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="출력 폴더 — train.yaml, holdout.yaml, report.json")
    p.add_argument("--n-train", type=int, default=16)
    p.add_argument("--n-holdout", type=int, default=4)
    p.add_argument("--distances", type=float, nargs="+", default=[0.20, 0.24, 0.28], help="카메라→보드 중심 [m]")
    p.add_argument("--tilts", type=float, nargs="+", default=[0, 15, 30, 45], help="보드 법선 대비 시선 기울기 [deg]")
    p.add_argument("--azimuth-step", type=float, default=30)
    p.add_argument("--roll-step", type=float, default=30)
    p.add_argument("--margin-px", type=float, default=25, help="보드 바깥 모서리 ↔ 영상 가장자리 최소 여유")
    p.add_argument("--min-square-px", type=float, default=14)
    p.add_argument("--clearance-m", type=float, default=0.06, help="link3~6·카메라 원점의 최소 높이(테이블 위)")
    p.add_argument("--max-seed-dev-deg", type=float, default=40,
                   help="관절 1~5 가 가장 가까운 교시 자세에서 벗어나도 되는 최대 각도 (j6 은 자유)")
    p.add_argument("--min-j5-deg", type=float, default=10, help="손목 뒤집힘 방지 — j5 최소값")
    p.add_argument("--max-reach-m", type=float, default=0.30, help="joint2 축 ↔ 손목 수평 거리 상한 (중력 부하)")
    p.add_argument("--start", type=float, nargs=6, default=[0, 1.7, -1.7, 0, 22, 0], help="정렬 시작 관절값[deg]")
    p.add_argument("--holdout-novelty-deg", type=float, nargs=2, default=[10, 25],
                   help="검증 자세 ↔ 가장 가까운 학습 자세의 손목 회전 차이 범위")
    p.add_argument("--keep-train", help="이미 쓴 학습 자세 YAML — 유지하고 --replace 번호만 새로 고른다(번호·순서 유지)")
    p.add_argument("--replace", default="", help="--keep-train 에서 바꿀 번호(0부터, 쉼표) — 손목 회전이 가장 비슷한 시점으로")
    p.add_argument("--evaluate", action="store_true", help="교시 보정 자세와 hand-eye 오차 비교(가상 잡음)")
    p.add_argument("--config", default=str(ROOT / "workcell/scans/piper_d405.yaml"))
    p.add_argument("--calibration", default=str(ROOT / "workcell/calibration/piper_d405.yaml"))
    args = p.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    cal = yaml.safe_load(Path(args.calibration).read_text())
    fk = PiperFK(ROOT / cfg["urdf"], cfg["base_frame"], cfg["tool_frame"])
    ffk = FastFK(fk)
    X = transform(cal["tool_T_color"])
    B = transform(cal["base_T_board"])
    K = np.asarray(cal["identity"]["color"]["K"], dtype=float)
    size = (int(cal["identity"]["color"]["width"]), int(cal["identity"]["color"]["height"]))
    centre = B[:3, :3] @ [BOARD_M[0] / 2, BOARD_M[1] / 2, 0] + B[:3, 3]
    table_z = float(B[2, 3])
    lo = ffk.lower + np.radians(LIMIT_MARGIN_DEG)
    hi = ffk.upper - np.radians(LIMIT_MARGIN_DEG)

    # IK 시드: 보정 원본의 교시 자세(보드를 보던 팔 모양) — 비슷한 팔 모양의 해가 나온다
    sample_dir = ROOT / cal["sample_directory"]
    samples = json.loads((sample_dir / "samples.json").read_text())
    seeds = [np.asarray(s["q"]) for s in samples["train"] + samples["holdout"]]
    seed_cams = [ffk.pose(q) @ X for q in seeds]

    rolls = list(np.arange(-180, 180, args.roll_step))
    azimuths = list(np.arange(0, 360, args.azimuth_step))
    feasible, tried = [], 0
    for c in candidates(centre, args.distances, args.tilts, azimuths, rolls):
        tried += 1
        cam = c["base_T_color"]
        order = np.argsort([np.linalg.norm(sc[:3, 3] - cam[:3, 3]) + 0.1 * np.radians(rot_deg(sc, cam))
                            for sc in seed_cams])[:4]
        q = solve_ik(ffk, cam @ np.linalg.inv(X), [seeds[i] for i in order], lo, hi)
        if q is None:
            continue
        # 교시로 이미 안전하게 다녀 본 팔 모양 근처만 — 손목 뒤집힘(j5<0)·낯선 팔꿈치 모양을 뺀다
        dev = min(np.degrees(np.max(np.abs(q[:5] - s[:5]))) for s in seeds)
        if dev > args.max_seed_dev_deg or np.degrees(q[4]) < args.min_j5_deg:
            continue
        frames = ffk.frames(q)
        if min(T[2, 3] for T in frames[2:]) - table_z < args.clearance_m or cam[2, 3] - table_z < args.clearance_m:
            continue
        if wrist_reach(frames) > args.max_reach_m:
            continue
        view = board_view(np.linalg.inv(cam) @ B, K, size, args.margin_px)
        if view is None or view["min_square_px"] < args.min_square_px:
            continue
        feasible.append({**{k: c[k] for k in ("distance_m", "view_tilt_deg", "azimuth_deg", "roll_deg")},
                         "q": q, "tool": frames[-1], "view": view})
    print(f"시점 {tried}개 중 IK·한계·시야·높이 통과 {len(feasible)}개")
    if args.keep_train:
        raw = [ln for ln in Path(args.keep_train).read_text().splitlines() if ln.startswith("- [")]
        qs = np.radians(yaml.safe_load(Path(args.keep_train).read_text())["poses_deg"])
        train = [{"q": q, "tool": ffk.pose(q), "line": ln,
                  "view": board_view(np.linalg.inv(ffk.pose(q) @ X) @ B, K, size, 0)} for q, ln in zip(qs, raw)]
        for i in [int(s) for s in args.replace.split(",") if s.strip()]:
            others = [t for j, t in enumerate(train) if j != i]
            pool = [c for c in feasible if min(rot_deg(c["tool"], t["tool"]) for t in others) > 5]
            new = min(pool, key=lambda c: rot_deg(c["tool"], train[i]["tool"]))
            print(f"#{i} 교체: 손목 회전 차이 {rot_deg(new['tool'], train[i]['tool']):.1f}°, "
                  f"뻗음 {wrist_reach(ffk.frames(train[i]['q'])) * 100:.1f} → {wrist_reach(ffk.frames(new['q'])) * 100:.1f} cm")
            train[i] = new
    else:
        if len(feasible) < args.n_train + args.n_holdout:
            raise SystemExit("통과한 시점이 부족합니다 — 거리·기울기 범위를 넓혀 보세요")
        train = order_by_motion(pick_diverse(feasible, args.n_train), np.radians(args.start))
    rest = [c for c in feasible if min(rot_deg(c["tool"], t["tool"]) for t in train) > 1]
    holdout = pick_holdout(rest, args.n_holdout, train, args.holdout_novelty_deg)
    if len(holdout) < args.n_holdout:
        raise SystemExit(f"검증 자세가 {len(holdout)}개뿐입니다 — --holdout-novelty-deg 범위를 넓혀 보세요")
    holdout = order_by_motion(holdout, train[-1]["q"])

    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    src = f"보정 {Path(args.calibration).name} ({cal.get('created_at', '?')}) 의 보드 위치·장착값으로 생성"
    write_yaml(out / "train.yaml", train, f"보정 학습 자세 {len(train)}개 — {src}")
    write_yaml(out / "holdout.yaml", holdout, f"보정 별도 검증 자세 {len(holdout)}개 — 학습과 다른 손목 회전. {src}")

    def summary(items):
        T = np.array([c["tool"] for c in items])
        rv = Rotation.from_matrix(T[0, :3, :3].T @ T[:, :3, :3]).as_rotvec()
        return {"count": len(items),
                "rotation_spread_singular_deg": np.degrees(np.linalg.svd(rv, compute_uv=False)).round(1).tolist(),
                "tilt_deg": sorted(round(c["view"]["tilt_deg"]) for c in items),
                "min_square_px": round(min(c["view"]["min_square_px"] for c in items), 1),
                "max_joint_step_deg": round(float(max(np.degrees(np.max(np.abs(a["q"] - b["q"])))
                                                      for a, b in zip(items, items[1:]))), 1) if len(items) > 1 else 0}
    report = {"board_centre_m": centre.round(4).tolist(), "candidates": tried, "feasible": len(feasible),
              "train": summary(train), "holdout": summary(holdout)}
    taught = [np.asarray(s["q"]) for s in samples["train"] if not s.get("excluded")]
    taught_T = np.array([ffk.pose(q) for q in taught])
    rv = Rotation.from_matrix(taught_T[0, :3, :3].T @ taught_T[:, :3, :3]).as_rotvec()
    report["taught_rotation_spread_singular_deg"] = np.degrees(np.linalg.svd(rv, compute_uv=False)).round(1).tolist()
    if args.evaluate:
        report["hand_eye_error_generated"] = hand_eye_error(ffk, [c["q"] for c in train], X, B, 300)
        report["hand_eye_error_taught"] = hand_eye_error(ffk, taught, X, B, 300)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"→ {out.relative_to(ROOT)}/train.yaml, holdout.yaml")


if __name__ == "__main__":
    main()

"""스캔 폴더의 규약 — 계획·프레임·복원 결과를 무엇으로, 어떤 이름으로 저장하나.

    data/{object}/scan/{name}/          (config.get_scan_dir)
      plan.npz, plan.json               ScanPlan — 스캔 자세와 그 근거, 촬영 당시의 물체 pose·장애물
      frames/frame_###.npz              촬영 한 장 — FRAME_KEYS
      trajectory.csv, trajectory.npz    장 사이를 잇는 실행 궤적 (scan_motion, 선택) — load_motion
      capture.json                      촬영 요약 (scan_capture 가 쓴다)
      recon{_tag}.ply, recon{_tag}.json 복원 메시(world 좌표)와 그 설정·평가

쓰는 쪽(Isaac 의 scan_capture)과 읽는 쪽(복원·뷰어)이 키 이름을 각자 알지 않도록 여기만
고친다. numpy 만 쓴다 — Isaac 의 Python 도 import 한다(trimesh 는 메시 입출력 함수 안에서만).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common import config, scene_config
from core.reconstruction.sensor import ScanCameraSpec

# 촬영 한 장(frame_###.npz)에 들어가는 배열.
#   depth        (H, W) float32 m — 광축 거리, 미적중 = inf
#   K            (3, 3) 내부 파라미터
#   world_T_cam  (4, 4) 렌더에 쓴 카메라 자세(OpenCV 광학 프레임) — 실물에선 FK + hand-eye
#   q_cmd/q_meas (6,)   명령한 관절값 / 촬영 순간 읽은 관절값
FRAME_KEYS = ("depth", "K", "world_T_cam", "q_cmd", "q_meas")


def scan_dir(object_name: str, name: str = "default") -> Path:
    return config.get_scan_dir(object_name, name)


@dataclass
class ScanPlan:
    """스캔 자세 N개와 그 근거. world_T_cam 은 IK 해의 FK(목표 goal_T_cam 과 수 mm 다를 수 있다).

    object_pose·obstacles 는 **계획 당시의 셀 상태**다. 분할·ICP·평가가 전역 config(활성 씬, 씬의
    물체 배치)가 아니라 이 값을 쓰므로, 다른 씬으로 띄운 Studio 나 기즈모로 옮긴 물체에서도 같은
    점이 지워진다. 옛 계획에는 없다(None) — 그때만 config 로 돌아간다.
    """

    q: np.ndarray                    # (N, 6) 관절값, 촬영 순서
    world_T_cam: np.ndarray          # (N, 4, 4)
    goal_T_cam: np.ndarray           # (N, 4, 4) IK 에 준 목표
    view_dirs: np.ndarray            # (N, 3) ROI 중심 → 카메라 단위벡터
    roll_deg: np.ndarray             # (N,) 광축 roll
    roi_center: np.ndarray           # (3,)
    roi_radius: float
    standoff: float                  # 카메라 ↔ ROI 중심 거리 (m)
    camera: ScanCameraSpec
    meta: dict = field(default_factory=dict)   # object, scene, roi_source, 후보 수 등 — 사람이 읽는 것
    object_pose: np.ndarray | None = None      # (7,) [x,y,z, qw,qx,qy,qz] world — 물체 바닥 중심
    obstacles: np.ndarray | None = None        # (M,10) [pos3, quat_wxyz4, dims3] 씬 장애물 OBB
    obstacle_names: list = field(default_factory=list)


def config_object_pose() -> np.ndarray:
    """지금 config.TARGET_OBJECT 의 pose (7,) — apply_object_placement / 덮어쓰기 뒤에 부른다."""
    return np.concatenate([np.asarray(config.TARGET_OBJECT["position"], dtype=np.float64),
                           np.asarray(config.TARGET_OBJECT["rotation"], dtype=np.float64)])


def snapshot_obstacles() -> tuple[np.ndarray, list]:
    """지금 config.OBSTACLES 의 OBB (M,10) 와 이름 — 플래너·Isaac 이 쓰는 것과 같은 OBB."""
    rows, names = [], []
    for obj in config.OBSTACLES:
        pos, quat, dims = scene_config.obstacle_obb(obj)
        rows.append(np.concatenate([pos, quat, dims]).astype(np.float64))
        names.append(obj["name"])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 10), names


def save_plan(directory: Path, plan: ScanPlan) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    extra = {}
    if plan.object_pose is not None:
        extra["object_pose"] = plan.object_pose
    if plan.obstacles is not None:
        extra["obstacles"] = plan.obstacles
    np.savez(directory / "plan.npz", q=plan.q, world_T_cam=plan.world_T_cam,
             goal_T_cam=plan.goal_T_cam, view_dirs=plan.view_dirs, roll_deg=plan.roll_deg,
             roi_center=plan.roi_center, roi_radius=plan.roi_radius, standoff=plan.standoff,
             **extra)
    record = {**plan.meta, "roi_center": np.asarray(plan.roi_center).tolist(),
              "roi_radius": float(plan.roi_radius), "standoff": float(plan.standoff),
              "n_reachable": int(len(plan.q)), "camera": plan.camera.to_dict()}
    # npz 가 원본이고 json 은 사람이 읽는 사본이다.
    if plan.object_pose is not None:
        record["object_pose"] = np.asarray(plan.object_pose).tolist()
    if plan.obstacles is not None:
        record["obstacles"] = [
            {"name": n, "pos": r[:3].tolist(), "quat": r[3:7].tolist(), "dims": r[7:].tolist()}
            for n, r in zip(plan.obstacle_names or [""] * len(plan.obstacles), plan.obstacles)]
    (directory / "plan.json").write_text(json.dumps(record, indent=2) + "\n")


MOTION_CSV = "trajectory.csv"


def load_motion(directory: Path):
    """장 사이 실행 궤적 → (joints (M,6), times (M,), shot_rows (N,)) 또는 없으면 None.

    shot_rows[i] = 장 i 의 자세가 놓인 행. 계획을 새로 쓸 때 scan_plan 이 옛 궤적을 지운다
    (clear_motion) — 호출자는 행 수·관절값이 계획과 맞는지도 본다(scan_capture).
    """
    path = (Path(directory) / MOTION_CSV).with_suffix(".npz")
    if not path.exists():
        return None
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        rows = meta.get("shot_rows")
        if rows is None:
            return None
        return (np.asarray(z["joints"], dtype=np.float64), np.asarray(z["times"], dtype=np.float64),
                np.asarray(rows, dtype=np.int64))


def clear_motion(directory: Path) -> None:
    """낡은 실행 궤적을 지운다 — 계획을 새로 쓸 때."""
    for suffix in (".csv", ".npz"):
        (Path(directory) / MOTION_CSV).with_suffix(suffix).unlink(missing_ok=True)


def load_plan(directory: Path) -> ScanPlan:
    d = np.load(directory / "plan.npz")
    meta = json.loads((directory / "plan.json").read_text())
    camera = ScanCameraSpec.from_dict(meta.pop("camera"))
    names = [o.get("name", "") for o in meta.pop("obstacles", None) or []]
    for k in ("roi_center", "roi_radius", "standoff", "n_reachable", "object_pose"):
        meta.pop(k, None)
    return ScanPlan(q=d["q"], world_T_cam=d["world_T_cam"],
                    goal_T_cam=d["goal_T_cam"] if "goal_T_cam" in d.files else d["world_T_cam"],
                    view_dirs=d["view_dirs"], roll_deg=d["roll_deg"], roi_center=d["roi_center"],
                    roi_radius=float(d["roi_radius"]), standoff=float(d["standoff"]),
                    camera=camera, meta=meta,
                    object_pose=d["object_pose"] if "object_pose" in d.files else None,
                    obstacles=d["obstacles"] if "obstacles" in d.files else None,
                    obstacle_names=names)


def frames_dir(directory: Path) -> Path:
    return directory / "frames"


def clear_frames(directory: Path) -> Path:
    """새 촬영 전에 그 run 의 낡은 산출물을 지운다 — 프레임, capture.json, 복원(recon*).

    복원은 프레임에서 나온 것이라 프레임을 새로 찍으면 더 이상 그 프레임의 복원이 아니다.
    """
    out = frames_dir(directory)
    out.mkdir(parents=True, exist_ok=True)
    stale = [*out.glob("frame_*.npz"), directory / "capture.json",
             *directory.glob("recon*.ply"), *directory.glob("recon*.json")]
    for old in stale:
        if old.exists():
            old.unlink()
    return out


def save_frame(directory: Path, index: int, **arrays) -> None:
    missing = set(FRAME_KEYS) - set(arrays)
    if missing:
        raise ValueError(f"frame {index}: missing {sorted(missing)}")
    np.savez_compressed(frames_dir(directory) / f"frame_{index:03d}.npz",
                        **{k: arrays[k] for k in FRAME_KEYS})


def list_runs(object_name: str, data_root: Path | None = None) -> list[str]:
    """data/{object}/scan/ 아래 계획이 있는 run 이름들."""
    base = (Path(data_root) / object_name / "scan") if data_root else config.get_scan_dir(object_name).parent
    return sorted(p.parent.name for p in base.glob("*/plan.npz"))


def list_recon_tags(directory: Path) -> list[str]:
    """그 run 의 복원 태그들 — "" (= recon.ply) 이 맨 앞."""
    tags = [p.stem[len("recon_"):] if p.stem.startswith("recon_") else ""
            for p in directory.glob("recon*.ply") if p.stem == "recon" or p.stem.startswith("recon_")]
    return sorted(tags, key=lambda t: (t != "", t))


def load_frame_poses(directory: Path) -> tuple[list[str], np.ndarray]:
    """프레임 이름과 world_T_cam (N,4,4) 만 — depth 를 풀지 않아 빠르다(npz 는 멤버별 압축)."""
    names, poses = [], []
    for f in sorted(frames_dir(directory).glob("frame_*.npz")):
        with np.load(f) as d:
            poses.append(d["world_T_cam"])
        names.append(f.stem)
    return names, np.asarray(poses).reshape(-1, 4, 4)


def load_frames(directory: Path) -> list[dict]:
    """[{name, depth, K, world_T_cam, q_cmd, q_meas}, ...] — 파일 이름 순(= 촬영 순)."""
    frames = []
    for f in sorted(frames_dir(directory).glob("frame_*.npz")):
        d = np.load(f)
        frames.append({"name": f.stem, **{k: d[k] for k in d.files}})
    if not frames:
        raise FileNotFoundError(f"no frames under {frames_dir(directory)} — scan_capture.py 를 먼저 돌린다")
    return frames


def recon_stem(tag: str = "") -> str:
    return f"recon_{tag}" if tag else "recon"


def save_recon(directory: Path, mesh, report: dict, tag: str = "") -> Path:
    """복원 메시(trimesh, world 좌표)와 설정·평가 → recon{_tag}.ply / .json."""
    path = directory / f"{recon_stem(tag)}.ply"
    mesh.export(str(path))
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return path


def load_recon(directory: Path, tag: str = ""):
    """복원 메시(trimesh) — 없으면 None."""
    path = directory / f"{recon_stem(tag)}.ply"
    if not path.exists():
        return None
    import trimesh

    return trimesh.load(str(path), force="mesh")

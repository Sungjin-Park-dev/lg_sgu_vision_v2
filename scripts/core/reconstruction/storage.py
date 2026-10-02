"""스캔 폴더의 규약 — 계획·프레임·복원 결과를 무엇으로, 어떤 이름으로 저장하나.

    data/{object}/scan/{name}/          (config.get_scan_dir)
      plan.npz, plan.json               ScanPlan — 스캔 자세와 그 근거
      frames/frame_###.npz              촬영 한 장 — FRAME_KEYS
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

from common import config
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
    """스캔 자세 N개와 그 근거. world_T_cam 은 IK 해의 FK(목표 goal_T_cam 과 수 mm 다를 수 있다)."""

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


def save_plan(directory: Path, plan: ScanPlan) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "plan.npz", q=plan.q, world_T_cam=plan.world_T_cam,
             goal_T_cam=plan.goal_T_cam, view_dirs=plan.view_dirs, roll_deg=plan.roll_deg,
             roi_center=plan.roi_center, roi_radius=plan.roi_radius, standoff=plan.standoff)
    (directory / "plan.json").write_text(json.dumps({
        **plan.meta, "roi_center": np.asarray(plan.roi_center).tolist(),
        "roi_radius": float(plan.roi_radius), "standoff": float(plan.standoff),
        "n_reachable": int(len(plan.q)), "camera": plan.camera.to_dict(),
    }, indent=2) + "\n")


def load_plan(directory: Path) -> ScanPlan:
    d = np.load(directory / "plan.npz")
    meta = json.loads((directory / "plan.json").read_text())
    camera = ScanCameraSpec.from_dict(meta.pop("camera"))
    for k in ("roi_center", "roi_radius", "standoff", "n_reachable"):
        meta.pop(k, None)
    return ScanPlan(q=d["q"], world_T_cam=d["world_T_cam"],
                    goal_T_cam=d["goal_T_cam"] if "goal_T_cam" in d.files else d["world_T_cam"],
                    view_dirs=d["view_dirs"], roll_deg=d["roll_deg"], roi_center=d["roi_center"],
                    roi_radius=float(d["roi_radius"]), standoff=float(d["standoff"]),
                    camera=camera, meta=meta)


def frames_dir(directory: Path) -> Path:
    return directory / "frames"


def clear_frames(directory: Path) -> Path:
    out = frames_dir(directory)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("frame_*.npz"):
        old.unlink()
    return out


def save_frame(directory: Path, index: int, **arrays) -> None:
    missing = set(FRAME_KEYS) - set(arrays)
    if missing:
        raise ValueError(f"frame {index}: missing {sorted(missing)}")
    np.savez_compressed(frames_dir(directory) / f"frame_{index:03d}.npz",
                        **{k: arrays[k] for k in FRAME_KEYS})


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

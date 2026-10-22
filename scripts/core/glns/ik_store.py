"""Shared on-disk store for per-viewpoint IK candidate sets.

한 곳(Check and Save IK, isaac_pipeline)에서 계산해 저장한 IK 후보를 다른 곳(Generate
Trajectory, glns/solve.py)이 그대로 재사용하기 위한 공유 포맷/경로/검증이다. 저장 대상은
solve.py 가 GTSP 에 넣기 직전의 원시 후보(증강→IK→dedup→충돌필터 이후)와 같은 모양이다.

경로는 ``data/{object}/ik/{N}/`` (mesh·viewpoint·trajectory 와 같은 층) 밑에 옵션
(roll/tilt/dedup)을 파일명으로 반영해 모드끼리 덮어쓰지 않게 하고(``ik/{N}/<...>.h5``),
유효성은 파일 안 attrs 로 판정한다 — **물체 pose 와 증강·dedup·IK 설정이 전부 일치**해야
재사용하고, 다르면 miss 로 보고 재계산한다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

STORE_FORMAT = "ik_solutions"
STORE_FORMAT_VERSION = 1

# _solve_pose_variant_candidates 가 뷰포인트마다 내놓는 metadata dict 의 키.
_META_SCALAR_KEYS = ("roll_deg", "tilt_deg", "tilt_azimuth_deg")
_META_VEC_KEYS = {"target_position": 3, "target_quaternion": 4}
_VARIANT_DTYPE = "U16"

# 검증 대상 설정과 그 비교 방식. object pose 와 IK/증강/dedup 파라미터가 전부 일치해야
# 재사용한다(_settings_match).
_FLOAT_ATOL = 1e-6


def augmentation_suffix(*, roll_augment: bool, tilt_augment: bool, dedup: bool) -> str:
    """옵션 조합을 파일명 접미사로 — 모드가 서로 다른 파일을 갖도록."""
    aug = "".join(part for part, on in (("roll", roll_augment), ("tilt", tilt_augment)) if on)
    name = aug or "nominal"
    return name if dedup else f"{name}_nodedup"


def ik_solutions_path(source_viewpoints, *, roll_augment: bool, tilt_augment: bool,
                      dedup: bool) -> Path:
    """저장 IK 경로 — ``data/{object}/ik/{N}/<옵션>.h5`` (mesh·viewpoint·trajectory 와 같은 층).

    소스 ``data/{object}/viewpoint/{N}/viewpoints_*.h5`` 에서 object 루트와 N 을 딴다.
    """
    src = Path(source_viewpoints)
    n_dir = src.parent                       # data/{object}/viewpoint/{N}
    object_root = n_dir.parent.parent        # data/{object}
    suffix = augmentation_suffix(roll_augment=roll_augment, tilt_augment=tilt_augment,
                                 dedup=dedup)
    return object_root / "ik" / n_dir.name / f"{suffix}.h5"


def build_settings(*, object_position, object_quat_wxyz, working_distance_m,
                   roll_augment, roll_step_deg, tilt_augment, tilt_angles_deg,
                   tilt_azimuths, dedup, dedup_rad, num_seeds, ik_seed,
                   lock_nominal_wrist3) -> dict:
    """저장/검증에 쓰는 정규화된 결과-영향 설정 dict — 두 호출자(check_ik / solve.py)가
    동일 dict 를 만들도록 한 곳에 모은다. batch_size 는 결과에 영향이 없어 제외한다(재사용
    조건에서 빼야 불필요한 miss 가 없다). dedup 이 꺼지면 dedup_rad 는 -1(무효)로 정규화해
    두 쪽이 항상 같은 값을 갖게 한다."""
    return {
        "object_position": np.asarray(object_position, dtype=np.float64).reshape(3),
        "object_quat_wxyz": np.asarray(object_quat_wxyz, dtype=np.float64).reshape(4),
        "working_distance_m": float(working_distance_m),
        "roll_augment": bool(roll_augment),
        "roll_step_deg": float(roll_step_deg),
        "tilt_augment": bool(tilt_augment),
        "tilt_angles_deg": np.asarray(tilt_angles_deg, dtype=np.float64).reshape(-1),
        "tilt_azimuths": int(tilt_azimuths),
        "dedup": bool(dedup),
        "dedup_rad": float(dedup_rad) if dedup else -1.0,
        "num_seeds": int(num_seeds),
        "ik_seed": int(ik_seed),
        "lock_nominal_wrist3": bool(lock_nominal_wrist3),
    }


def _settings_match(want: dict, got: dict) -> bool:
    for key, wv in want.items():
        gv = got.get(key)
        if gv is None:
            return False
        if key in ("object_position", "object_quat_wxyz", "tilt_angles_deg"):
            wa, ga = np.asarray(wv, np.float64), np.asarray(gv, np.float64)
            if wa.shape != ga.shape or not np.allclose(wa, ga, atol=_FLOAT_ATOL):
                # quaternion 은 부호 반전(q, -q)이 같은 회전이라 한 번 더 본다.
                if key == "object_quat_wxyz" and np.allclose(wa, -ga, atol=_FLOAT_ATOL):
                    continue
                return False
        elif isinstance(wv, float):
            if not np.isclose(wv, float(gv), atol=1e-9, rtol=1e-6):
                return False
        else:
            if wv != type(wv)(gv):
                return False
    return True


def _concat(parts: list, ncol: int | None) -> np.ndarray:
    shaped = [np.asarray(p).reshape(-1, ncol) if ncol else np.asarray(p).reshape(-1)
              for p in parts]
    if not shaped:
        return np.empty((0, ncol) if ncol else (0,), dtype=np.float64)
    return np.concatenate(shaped, axis=0)


def save_ik_solutions(path, representatives, metadata, settings: dict, *,
                      source_viewpoints, object_name) -> int:
    """원시 IK 후보(collision filter 이후) + 설정을 h5 로 원자적으로 쓴다. 저장된 후보 수 반환."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.asarray([len(r) for r in representatives], dtype=np.int32)
    joints = _concat([np.asarray(r, dtype=np.float64).reshape(-1, 6)
                      for r in representatives], 6)
    variant = _concat([m["variant"] for m in metadata], None).astype("S16")
    reachable = counts > 0
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["format"] = STORE_FORMAT
        f.attrs["format_version"] = STORE_FORMAT_VERSION
        f.attrs["object"] = str(object_name)
        f.attrs["source_viewpoints"] = str(source_viewpoints)
        f.attrs["num_viewpoints"] = int(len(counts))
        f.attrs["reachable_count"] = int(reachable.sum())
        f.attrs["created_at"] = datetime.now().isoformat()
        for key, val in settings.items():
            f.attrs[key] = val
        f.create_dataset("success_counts", data=counts)
        f.create_dataset("reachable", data=reachable)
        f.create_dataset("joints", data=joints)
        f.create_dataset("meta_variant", data=variant)
        for key in _META_SCALAR_KEYS:
            f.create_dataset(f"meta_{key}", data=_concat([m[key] for m in metadata], None))
        for key, ncol in _META_VEC_KEYS.items():
            f.create_dataset(f"meta_{key}", data=_concat([m[key] for m in metadata], ncol))
    tmp.replace(path)
    return int(counts.sum())


def load_ik_solutions(path, want_settings: dict) -> tuple[list, list] | None:
    """설정이 전부 일치하면 (representatives, metadata) 를, 아니면(없음/포맷/설정 불일치/손상) None.

    반환 모양은 ``_solve_pose_variant_candidates`` + ``_collision_filter_representatives``
    의 결과와 동일하다 — solve.py 가 그대로 이어받아 GTSP 로 넘길 수 있다.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with h5py.File(path, "r") as f:
            if f.attrs.get("format") != STORE_FORMAT:
                return None
            got = {}
            for key in want_settings:
                if key not in f.attrs:
                    return None
                got[key] = f.attrs[key]
            if not _settings_match(want_settings, got):
                return None
            counts = np.asarray(f["success_counts"], dtype=np.int32)
            joints = np.asarray(f["joints"], dtype=np.float64)
            variant = np.asarray(f["meta_variant"]).astype(_VARIANT_DTYPE)
            scalars = {k: np.asarray(f[f"meta_{k}"], dtype=np.float64)
                       for k in _META_SCALAR_KEYS}
            vecs = {k: np.asarray(f[f"meta_{k}"], dtype=np.float64)
                    for k in _META_VEC_KEYS}
    except (OSError, KeyError):
        return None

    total = int(counts.sum())
    if joints.shape != (total, 6) or variant.shape[0] != total:
        return None
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    representatives, metadata = [], []
    for vp in range(len(counts)):
        a, b = int(offsets[vp]), int(offsets[vp + 1])
        representatives.append(joints[a:b].reshape(-1, 6).copy())
        entry = {"variant": variant[a:b].astype(_VARIANT_DTYPE).copy()}
        for key in _META_SCALAR_KEYS:
            entry[key] = scalars[key][a:b].copy()
        for key, ncol in _META_VEC_KEYS.items():
            entry[key] = vecs[key][a:b].reshape(-1, ncol).copy()
        metadata.append(entry)
    return representatives, metadata

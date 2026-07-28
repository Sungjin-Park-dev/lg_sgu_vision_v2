#!/usr/bin/env python3
"""data/ 를 역할별 3폴더 + 통일된 파일명으로 이관한다 (1회성).

앱마다 다른 규칙으로 자란 산출물을 정리한다:

  data/{object}/ik/{N}/glns_result_{studio,gui,…}.h5  →  trajectory/{N}/solution.h5
  trajectory/{N}/glns_trajectory_joined.*             →  trajectory/{N}/trajectory.*
  trajectory/{N}/glns_trajectory_home_to_start.*      →  trajectory_home_to_start.*
  trajectory/{N}/glns_trajectory_comp*.{csv,npz}      →  삭제 (더 이상 생성 안 함)
  trajectory/{N}/trajectory_dp_ee_….{csv,npz}         →  삭제 (DP 제거)
  trajectory/{N}/ans_*.csv                            →  삭제
  placement_sweep/{summary.*,heatmap_*.png}           →  docs/reference/placement-sweep/{object}/
  placement_sweep/ 나머지(스윕 raw)                     →  삭제

남는 것은 mesh / viewpoint / trajectory 세 폴더뿐이다.

**기본은 dry-run** — 되돌릴 수 없는 삭제가 섞여 있어 목록을 눈으로 본 뒤 ``--apply`` 한다.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SWEEP_DOCS = PROJECT_ROOT / "docs" / "reference" / "placement-sweep"

SWEEP_KEEP = ("summary.csv", "summary.json")
SWEEP_KEEP_GLOB = "heatmap_*.png"


class Plan:
    """이관 작업 목록. dry-run 이면 출력만, --apply 면 실제로 수행한다."""

    def __init__(self, apply: bool):
        self.apply = apply
        self.moves: list[tuple[Path, Path]] = []
        self.deletes: list[Path] = []

    def move(self, src: Path, dst: Path) -> None:
        self.moves.append((src, dst))

    def delete(self, path: Path) -> None:
        self.deletes.append(path)

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    def run(self) -> int:
        for src, dst in self.moves:
            print(f"  MOVE   {self._rel(src)}\n      →  {self._rel(dst)}")
            if self.apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
        for path in self.deletes:
            kind = "DEL/d " if path.is_dir() else "DEL   "
            print(f"  {kind} {self._rel(path)}")
            if self.apply:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
        print(f"\n  이동 {len(self.moves)}건, 삭제 {len(self.deletes)}건"
              + ("" if self.apply else "  — dry-run (실행하려면 --apply)"))
        return 0


def _num_dirs(base: Path):
    """``{N}`` 형태의 하위 디렉토리만 (숫자 이름)."""
    if not base.is_dir():
        return []
    return sorted((d for d in base.iterdir() if d.is_dir() and d.name.isdigit()),
                  key=lambda d: int(d.name))


def migrate_object(obj_dir: Path, plan: Plan) -> None:
    obj = obj_dir.name
    traj_root = obj_dir / "trajectory"

    # --- 1) ik/{N}/glns_result_*.h5 → trajectory/{N}/solution.h5 (최신 1개만) -----
    for n_dir in _num_dirs(obj_dir / "ik"):
        results = sorted(n_dir.glob("glns_result*.h5"))
        if results:
            newest = max(results, key=lambda p: p.stat().st_mtime)
            plan.move(newest, traj_root / n_dir.name / "solution.h5")
            for extra in results:
                if extra != newest:
                    plan.delete(extra)     # 앱별 중복본(_studio/_gui/타임스탬프)
    if (obj_dir / "ik").is_dir():
        plan.delete(obj_dir / "ik")

    # --- 2~4) trajectory/{N} 안의 이름 통일과 폐기 --------------------------------
    renames = {
        "glns_trajectory_joined": "trajectory",
        "glns_trajectory_home_to_start": "trajectory_home_to_start",
        "glns_trajectory_end_to_home": "trajectory_end_to_home",
    }
    for n_dir in _num_dirs(traj_root):
        for old, new in renames.items():
            for suffix in (".csv", ".npz"):
                src = n_dir / f"{old}{suffix}"
                if src.exists():
                    plan.move(src, n_dir / f"{new}{suffix}")
        # 성분별 궤적: verify 가 더 이상 만들지 않는다(joined 만 남긴다)
        for stale in sorted(n_dir.glob("glns_trajectory_comp*")):
            plan.delete(stale)
        # DP 산출물: 백엔드 자체가 사라졌다
        for stale in sorted(n_dir.glob("trajectory_dp_*")):
            plan.delete(stale)
        # 어떤 코드도 읽지 않는 수동 사본
        for stale in sorted(n_dir.glob("ans_*")):
            plan.delete(stale)

    # --- 5) placement_sweep: 결정 근거만 docs 로, raw 는 폐기 ---------------------
    sweep = obj_dir / "placement_sweep"
    if sweep.is_dir():
        for name in SWEEP_KEEP:
            src = sweep / name
            if src.exists():
                plan.move(src, SWEEP_DOCS / obj / name)
        for src in sorted(sweep.glob(SWEEP_KEEP_GLOB)):
            plan.move(src, SWEEP_DOCS / obj / src.name)
        plan.delete(sweep)      # 남은 raw(glns_result_*.h5, solve_*.log)까지 통째로


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="실제로 이동/삭제한다 (기본은 dry-run)")
    parser.add_argument("--object", default=None,
                        help="한 물체만 처리 (기본: data/ 전체)")
    args = parser.parse_args()

    if not DATA_ROOT.is_dir():
        print(f"data/ 가 없다: {DATA_ROOT}")
        return 2

    objects = ([DATA_ROOT / args.object] if args.object
               else sorted(d for d in DATA_ROOT.iterdir() if d.is_dir()))
    missing = [d for d in objects if not d.is_dir()]
    if missing:
        print(f"없는 물체: {[d.name for d in missing]}")
        return 2

    plan = Plan(apply=args.apply)
    for obj_dir in objects:
        migrate_object(obj_dir, plan)

    print("=" * 70)
    print("DATA LAYOUT MIGRATION" + ("" if args.apply else "  [dry-run]"))
    print("=" * 70)
    if not plan.moves and not plan.deletes:
        print("  할 일 없음 — 이미 정리된 상태다.")
        return 0
    return plan.run()


if __name__ == "__main__":
    sys.exit(main())

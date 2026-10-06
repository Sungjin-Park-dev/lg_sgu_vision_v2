#!/usr/bin/env python3
"""복원 메시(world 좌표 recon.ply)를 물체 좌표의 OBJ 로 내보낸다.

지금은 **평가용 한 가지 방식**만 있다: 정답 배치(씬 object_placements)의 역변환을 씌워 정답
source.obj 와 같은 물체 좌표에 놓는다(sim.world_T_object). 그래서 복원 메시로 만든 viewpoint 를
정답 것과 그대로 비교할 수 있다. 정답 배치를 빌려 쓰는 셈이라 도면 없는 실제 흐름은 아니다 —
자기 좌표(바닥 중심 등)로 내보내는 방식은 파일 이름·물체 폴더 규약을 정한 뒤에 붙인다.

출력: data/{object}/mesh/recon.obj (Viewpoint Studio 의 Mesh 목록에 뜬다)

Usage:
    uv run scripts/core/reconstruction/export_recon.py --object curved_structure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config, scene_config  # noqa: E402
from core.reconstruction import sim, storage  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True)
    scene_config.add_cli_argument(p)
    p.add_argument("--name", default="default", help="data/{object}/scan/{name}")
    p.add_argument("--tag", default="", help="recon_{tag}.ply 를 내보낸다")
    p.add_argument("--out", default="recon.obj", help="data/{object}/mesh/ 아래 파일 이름")
    return p.parse_args()


def main():
    args = parse_args()
    scene_config.apply_cli(args, config)
    scan = storage.scan_dir(args.object, args.name)
    pose = sim.scan_object_pose(storage.load_plan(scan))
    # 계획에 스캔 당시 pose 가 있으면 그것을 쓴다. 옛 계획만 씬 배치에 기댄다.
    if pose is None and not config.apply_object_placement(args.object):
        sys.exit(f"'{args.object}' 의 배치가 계획에도 씬에도 없다 — 정답 좌표로 옮길 수 없다")

    recon = storage.load_recon(scan, args.tag)
    if recon is None:
        sys.exit("복원 메시가 없다 — reconstruct.py 를 먼저 돌린다")
    out, bbox = sim.export_gt_frame(recon, args.object, pose, args.out)
    print(f"[export] {len(recon.faces)} faces -> {out.relative_to(PROJECT_ROOT)}")
    for line in bbox.splitlines():
        print(f"[export]   {line}")


if __name__ == "__main__":
    main()

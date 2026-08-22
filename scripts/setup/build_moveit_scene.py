#!/usr/bin/env python3
"""씬 YAML → MoveIt/cuMotion 이 읽는 ``.scene`` 파일.

MoveIt 과 Inspection 이 **같은 셀**을 보게 하는 다리다. 지금까지 MoveIt/cuMotion 은
장애물이 하나도 없는 빈 월드로 계획해 왔다 — cuMotion 의 월드 입구가 둘인데 둘 다
꺼져 있었기 때문이다:
  - ESDF(nvblox): launch 에서 의도적으로 off (nvblox 를 안 쓴다)
  - StaticPlanningSceneServer: ``moveit_collision_objects_scene_file`` 이 빈 문자열
이 스크립트가 두 번째 입구를 채운다. 파일이 없으면 그 노드는 에러 없이 조용히
넘어가므로, 빈 월드로 계획하고 있다는 사실이 드러나지 않았다.

프레임: 씬 YAML 도 MoveIt 도 ``base_link`` 다(MoveIt 은 world==base_link). 그래서
좌표 변환이 없다 — 씬 값이 그대로 들어간다.

포맷(문서가 없어 cuMotion 파서에 직접 넣어보며 확정, 2026-08-22):
    <이름>+                     ← 헤더. **끝의 '+' 가 필수**다(없으면 "Missing header")
    * <물체 id>
    <x y z>                     ← cuMotion 은 이것을 primitive_pose 로 읽는다
    <qx qy qz qw>               ← **x y z w 순서**(쿼터니언 w 가 마지막)
    <shape 개수>
    <shape 종류>                ← box / sphere / cylinder 만 지원
    <치수>
    <shape x y z>               ← cuMotion 은 무시하지만 **줄은 있어야 한다**
    <shape qx qy qz qw>         ← 〃
    <r g b a>                   ← 〃
    <subframe 개수>             ← 이 줄이 빠지면 **다음 물체가 조용히 통째로 사라진다**
    .
마지막 항목이 특히 위험하다: 개수가 안 맞아도 success=True 로 돌아오고 물체만 없어진다.

사용법:
    uv run --no-sync scripts/setup/build_moveit_scene.py
    uv run --no-sync scripts/setup/build_moveit_scene.py --scene sim_default -o /tmp/x.scene
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import config, scene_config  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "scripts" / "moveit" / "cell.scene"
COLOR = (0.5, 0.5, 0.5, 1.0)


def object_block(name: str, pos, quat_wxyz, dims) -> str:
    """물체 하나의 10줄. quat 은 씬의 (w,x,y,z) → 파일의 (x,y,z,w) 로 재배열한다."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    f3 = lambda v: " ".join(f"{float(c):.6f}" for c in v)
    return "\n".join([
        f"* {name}",
        f3(pos),
        f"{x:.6f} {y:.6f} {z:.6f} {w:.6f}",
        "1",
        "box",
        f3(dims),
        "0 0 0",                 # shape pose (cuMotion 은 안 읽지만 줄은 필수)
        "0 0 0 1",
        " ".join(f"{c:.3f}" for c in COLOR),
        "0",                     # subframe 개수 — 빠지면 다음 물체가 사라진다
    ])


def build(scene_name: str) -> str:
    config.load_scene(scene_name)
    config.sync_support_to_target()      # support 기둥은 물체 배치에서 파생된다
    blocks = []
    for obstacle in config.OBSTACLES:
        pos, quat, dims = scene_config.obstacle_obb(obstacle)
        blocks.append(object_block(obstacle["name"], pos, quat, dims))
    return "\n".join([f"{config.ACTIVE_SCENE}+", *blocks, ".", ""])


def write_scene_file(scene_name: str, output_file) -> str:
    """씬 YAML → ``output_file`` 에 .scene 을 쓰고 그 경로를 돌려준다.

    launch 가 이걸 부른다. URDF 를 xacro 로 처리해 /tmp 에 덤프하고 그 경로를 cuMotion 에
    넘기는 기존 패턴과 같다 — 생성물을 리포에 두면 씬 YAML 을 고치고 재생성을 잊었을 때
    MoveIt 만 옛 셀을 보게 되는데, 그게 에러 없이 조용히 일어난다.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(build(scene_name))
    return str(output_file)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    scene_config.add_cli_argument(ap)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    text = build(args.scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    n = text.count("\n* ") + (1 if text.startswith("* ") else 0)
    print(f"[moveit-scene] {args.output.relative_to(PROJECT_ROOT)}: "
          f"{len(config.OBSTACLES)} obstacles, frame=base_link")
    for o in config.OBSTACLES:
        p, _, d = scene_config.obstacle_obb(o)
        print(f"    {o['name']:14s} pos={np.round(p,4)} dims={np.round(d,3)}")


if __name__ == "__main__":
    main()

# Isaac Sim 셋업

UR20 로봇을 Isaac Sim에서 시각화하고, ROS2 `/joint_states`를 받아 시뮬레이션 로봇이 실제 로봇 상태를 따라 움직이게 하는 환경 구성. `docs/setup_docker.md`의 컨테이너 셋업이 완료된 상태에서 진행.

## 역할

Isaac Sim은 이 프로젝트에서 **시각화/검증 용도**로 사용 — trajectory 계획이나 IK는 cuRobo가 담당. Isaac Sim은:

- UR20 로봇 USD를 로드하여 3D 시각화
- `/joint_states` 토픽 구독 → ArticulationController로 시뮬 로봇 동기화
- 워크셀(테이블, 벽, 검사 객체 등)도 함께 표시 (선택)

ur_robot_driver의 mock hardware나 URSim으로 명령된 로봇 상태를 그대로 미러링.

## 워크플로우 개요

```
[ur_robot_driver] → /joint_states → [Isaac Sim ArticulationController]
                                          ↓
                                   시뮬레이션 로봇 동기화
```

## 사전 준비

`pyproject.toml`에 `isaacsim[all,extscache]==6.0.0`이 있으니 `uv sync`로 설치됨. 별도 작업 불필요.

확인:
```bash
# 컨테이너, venv 활성 상태
python -c "import isaacsim; print('isaacsim OK')"
```

---

## URDF → USD 변환 (GUI URDF Importer)

Isaac Sim의 GUI URDF Importer 사용. CLI `urdf_usd_converter`보다 articulation root 등 옵션을 정확히 컨트롤할 수 있어 안정적.

### 1) 빈 Isaac Sim 띄우기

```bash
# 컨테이너, 셸 B (venv 활성, ROS2 sourcing X)
uv run scripts/isaac/launch_sim.py
```

`launch_sim.py`는 빈 stage + URDF Importer 확장만 켜진 상태로 진입.

### 2) URDF Importer 열기

상단 메뉴:

**Isaac Utils → Workflows → URDF Importer**

또는:

**File → Import → URDF**

### 3) URDF 파일 선택 + 옵션 설정

- **Input File**: `ur20_description/ur20_with_camera.urdf` 선택
- **Output Directory**: `ur20_description/ur20/` 같은 새 폴더
- **Joint Drive Type**: `Position`
- **Articulation Root**: `base_link` 또는 자동
- **Self Collision**: 보통 disabled
- **Merge Fixed Joints**: 적당히 활성화 (선호에 따라)

### 4) Import + Save

`Import` 클릭 → stage에 로봇이 로드됨.

`File → Save As`로 USD 저장:
```
ur20_description/ur20/ur20.usd
```

CLAUDE.md 또는 스크립트가 이 경로를 참조하므로 위치 중요.

### 5) Articulation Root 검증

Stage 트리에서 base_link prim 클릭 → 우측 Property 패널에서 **`PhysicsArticulationRootAPI`**가 적용되어 있는지 확인.

또는 Window → Script Editor에서:
```python
from pxr import UsdPhysics
import omni.usd
stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print("Articulation root:", prim.GetPath())
```

`base_link` 또는 그 부모에 articulation root가 있어야 함.

---

## joint_control.py 사용

생성한 USD를 로드하고 Action Graph로 ROS2 연동:

```bash
# 컨테이너, 셸 B
uv run scripts/isaac/joint_control.py
```

워크셀 포함:
```bash
uv run scripts/isaac/joint_control.py --object sample
```

`--object sample` 사용 시 `config.py`의 TABLE/WALLS/ROBOT_MOUNT/TARGET_OBJECT를 stage에 추가.

### 스크립트 동작

1. `ur20.usd` 로드 → `/World/UR20`
2. `--object` 지정 시 워크셀 cuboid + 타겟 메시 추가
3. ArticulationRoot 자동 탐지 (없으면 STAGE_PATH에 fallback 적용)
4. ROS2 bridge 확장 활성화
5. Action Graph 생성:
   - `OnPlaybackTick` → `ROS2SubscribeJointState` → `IsaacArticulationController`
   - 토픽: `/joint_states`
6. Action Graph 창 자동 열기
7. Physics 초기화 + Play

### 정상 동작 확인

콘솔 출력:
```
Articulation root: /World/UR20/base_link    ← 또는 비슷한 경로
```

`Available DOFs: []` 경고가 **없어야** 정상.

---

## pipeline_ui.py — 물체 선택·이동·궤적 생성/테스트

`joint_control.py`와 같은 워크셀을 띄우고, Omni.UI 3패널(Generate / Preview / Publish)로
**물체를 고르고, 뷰포트 기즈모로 옮긴 뒤, 그 위치에서 궤적을 생성·미리보기**한다.

```bash
uv run scripts/isaac/pipeline_ui.py --object sample
```

### 워크플로

1. **물체 선택** — Generate 패널 상단 `Object` 드롭다운에서 선택 → `Load Object`.
   `/World/target_object` prim이 기본 pose로 (재)생성되고 `--object` 필드가 동기화된다.
2. **위치 변경** — 뷰포트에서 물체 **루트 prim**(`/World/target_object`)을 선택하고
   `W`(이동)/`E`(회전) 기즈모로 드래그. (Stage 트리에서 루트를 선택하면 확실하다.)
3. **궤적 생성** — `Generate Trajectory`. 물체의 현재 월드 pose를 읽어 robot frame으로 변환
   (`z -= 0.805`) 후 `plan_trajectory.py`에 `--object-position`/`--object-quat`로 전달.
   stage에 물체가 없으면 생성을 막고 안내한다(stale pose 방지).
4. **테스트** — Preview 패널 `Load & Preview` → `Play`. ghost UR20가 옮긴 물체를 향해
   검사 궤적을 재생한다.

### sample 외 물체 (source.usd 선행 변환)

드롭다운은 `source.obj`가 있는 물체를 모두 보여주지만, 로드에는 `source.usd`가 필요하다.
`sample`만 기본 제공되므로 다른 물체는 1회 변환한다(`omni.kit.asset_converter`):

```bash
uv run scripts/isaac/usd/build_object_usd.py --object curved_structure
```

### 방향 보정 (STP→OBJ 회전 베이크)

**방향 규칙**: `config.TARGET_OBJECT["rotation"] = identity`. 즉 config는 회전을 적용하지 않고,
**물체 방향은 전부 메시(`source.obj`)에 베이크**한다. 그래야 **viser(로컬 프레임)와
Isaac(월드 프레임)이 회전 차이 없이 동일**하게 보인다("그냥 올리면 똑바로").

STP→OBJ 변환 시 메시에 박힌 이상한 방향은 `normalize_mesh.py`(스케일·재중심만)로는 안 고쳐진다.
Isaac에서 똑바로 안 서면:

1. UI에서 물체 로드 → 뷰포트에서 `E`(회전) 기즈모로 똑바로 세운다.
2. `Log Pose` 버튼 → 로그의 `--world-target-quat W X Y Z` 줄을 복사.
3. 베이크: `uv run scripts/prep/reorient_mesh.py --object {obj} --world-target-quat W X Y Z`
   (config가 identity라 target을 그대로 베이크). 단순 90° 뒤집기는 `--euler -90 0 0`도 가능.
4. `uv run scripts/isaac/usd/build_object_usd.py --object {obj} --force` → source.usd 재생성.
5. **viewpoint 재생성**(메시가 바뀌었으므로) → UI 재시작 → 기본 로드부터 똑바로 + viser와 동일.

> **sample**: 이전 90°는 순수 z-yaw였어서 identity로 바꿔도 **똑바로 선 채 90°만 풀림**.
> 기존 viewpoints는 그대로 유효(z회전 불변), trajectory만 재계획하면 됨. 특정 yaw로 두고
> 싶으면 sample 메시도 같은 방식으로 베이크(단, `data/sample` 파일이 root 소유면 chown 필요).

### 한계

- **viewpoints.h5는 mesh-local** → 평행이동은 재계획만으로 충분. 단, `generate_viewpoints`의
  bottom-facing 필터가 저장 전 회전을 반영하므로 **큰 회전은 viewpoint 재생성 권장**.
- cuRobo 충돌월드의 `support` 받침대는 원위치 고정 → **테이블 위 소폭 이동 범위**에서 사용.

---

## 동시 실행 흐름

### Mock hardware + Isaac Sim 시각화

```bash
# 셸 A: driver (시스템 ROS2)
source /opt/ros/jazzy/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur20 use_mock_hardware:=true launch_rviz:=true

# 셸 B: Isaac Sim (venv, 시스템 ROS2 X)
uv run scripts/isaac/joint_control.py

# 셸 C: 명령 전송 (시스템 ROS2 + venv)
source /opt/ros/jazzy/setup.bash
uv run scripts/ros2/move_to_start.py
uv run scripts/pipeline/publish_trajectory.py --csv data/sample/trajectory/124/trajectory.csv
```

셸 C에서 명령을 보내면 셸 A의 mock hardware가 즉시 state 반영 → 셸 B의 Isaac Sim이 따라 움직임.

### URSim + Isaac Sim 시각화

위에서 셸 A를 URSim 풀 모드로 바꿈. `docs/setup_ursim.md` 참조.

---

## launch_sim.py — 빈 시뮬레이터

URDF/USD를 GUI에서 수동 import하거나 stage를 자유롭게 만들 때 사용:

```bash
uv run scripts/isaac/launch_sim.py
```

- 로봇 로드 X
- Action Graph 생성 X
- ROS2 bridge + URDF Importer 확장만 활성

빈 stage에 직접 prim 추가하거나 USD 열기.

---

## 셸 환경 주의

Isaac Sim은 **자기 번들 ROS2 jazzy**를 사용. 시스템 `/opt/ros/jazzy`와 같은 셸에서 sourcing되면 `LD_LIBRARY_PATH`에서 FastDDS/FastCDR 우선순위 다툼 → ABI 충돌 위험.

### 권장: 별도 셸

| 셸 B (Isaac Sim 전용) | |
|---|---|
| `source /workspace/.venv/bin/activate` | venv 활성 |
| `source /opt/ros/jazzy/setup.bash` | **하지 말 것** |

`.venv/bin/activate`가 `LD_LIBRARY_PATH`에 isaacsim 번들 lib을 추가하도록 설정되어 있음. activate 후:
```bash
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep isaacsim
# /workspace/.venv/.../isaacsim.ros2.core/jazzy/lib 출력
```

다른 셸에서 시스템 ROS2를 source한 노드들과는 **같은 DOMAIN_ID + 같은 RMW(`rmw_fastrtps_cpp`)**라 통신 가능.

---

## 트러블슈팅

### `Available DOFs: []`

ArticulationRoot가 잘못된 prim에 적용됨. URDF Importer GUI로 재변환 권장.

런타임 임시 해결 (스크립트가 자동 처리):
```python
# fallback: STAGE_PATH에 ArticulationRootAPI 적용
UsdPhysics.ArticulationRootAPI.Apply(stage_prim)
```

근본 해결: USD 자체를 GUI로 다시 만들기.

### `OpenWindow` 명령 에러

```
[Error] [omni.kit.commands.command] Can't execute command: "OpenWindow"
```

Action Graph 확장이 아직 로드 중일 때 발생. 무해함 — 창은 결국 뜸. Window → Visual Scripting → Action Graph 메뉴로 수동 열기 가능.

### `ROS2 Bridge startup failed`

Isaac Sim 번들 ROS2 lib 경로 미설정. `.venv/bin/activate` 안에 다음이 있어야 함:
```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib"
```

### Isaac Sim 노드가 ROS2 그래프에 안 보임

다른 셸에서 `ros2 node list`에 Isaac Sim 노드 미출현. RMW/DOMAIN_ID 미스매치:
```bash
# Isaac Sim 셸
echo $RMW_IMPLEMENTATION    # rmw_fastrtps_cpp
echo $ROS_DOMAIN_ID         # 0 (또는 비어있음)

# 다른 셸 (시스템 ROS2)
echo $RMW_IMPLEMENTATION    # rmw_fastrtps_cpp
echo $ROS_DOMAIN_ID         # 0
```

양쪽 일치 필요.

### SSL Certificate Verify Failed

```
ssl.SSLCertVerificationError: ... unable to get local issuer certificate
```

`omni.kit.window.extensions`가 markdown URL verify할 때 발생. 무해 (시뮬 동작 무관).

깔끔히 없애려면:
```bash
apt install -y ca-certificates
update-ca-certificates
```

### `negative mass` 또는 `inertia tensor` 경고

URDF의 일부 link가 mass 정의 누락. PhysX가 작은 sphere로 추정해서 동작은 하지만 dynamics 부정확. URDF에 mass/inertia 명시하거나 무시 가능 (시각화 목적이면 무관).

### Articulation root가 base_link인데 DOF 0개

URDF Importer 변환 시 articulation 구조가 깨진 경우. 다시 import하면서:
- **Merge Fixed Joints** 옵션 토글
- **Self Collision** 비활성화
- **Articulation Root** 명시적 지정

저장 후 `joint_control.py` 재실행.

---

## 다음 단계

- 파이프라인 실행 (viewpoint 생성 → IK → publish): `docs/running.md`
- config 파라미터 (카메라, 워크셀 위치 등): `docs/config.md`

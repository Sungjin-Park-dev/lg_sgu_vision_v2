# PiPER + D405 수동 실제 스캔

손목에 고정한 RealSense D405로 작은 물체를 여러 방향에서 촬영하고, 기존 CAD-less TSDF 엔진으로 PLY·OBJ를 만든다. 이번 구현의 범위는 **카메라 장착 보정 → 수동 촬영 → 복원**이다. 자동 검사 궤적 생성·로봇 실행은 연결하지 않았다.

## 실행 구조

코드와 데이터는 호스트의 `lg_sgu_vision_v2`에 둔다. 기존 `ros2-jazzy-piper` 컨테이너에 프로젝트를 추가 마운트해서 ROS Jazzy·RealSense·PiPER SDK 환경으로 실행한다.

| 호스트 폴더 | PiPER 컨테이너 경로 | 용도 |
|---|---|---|
| `/home/sungjin/Documents/agx_arm_ws` | `/workspace/agx_arm_ws` | 기존 PiPER 드라이버·카메라 실행 파일 |
| `/home/sungjin/Documents/lg_sgu_vision_v2` | `/workspace/lg_sgu_vision_v2` | 스캔 앱·보정·촬영·복원 결과 |

기존 UR/Isaac Docker 구성과 로봇 프로필은 변경하지 않는다. 스캔 의존성은 별도 `.venv-piper`에 설치하며 기존 Isaac용 `.venv`를 활성화하지 않는다.

| 코드·설정 | 내용 |
|---|---|
| `scripts/apps/piper_scan.py` | Viser 앱: 보정·검증·촬영·복원·현재 카메라 표시 |
| `scripts/core/realscan/capture.py` | 관절 이력·정지 판정·시간 동기화, 카메라 프레임, 촬영 세션(저장) — 로봇 종류 무관 |
| `scripts/core/realscan/calibration.py` | ChArUco 검출, hand-eye 계산, 별도 자세 검증, 보드 depth 평면 검사 — 로봇 종류 무관 |
| `scripts/core/realscan/piper.py` | PiPER 전용: joint1~6, URDF FK, CAN 수신 전용 SDK(송신 차단) |
| `scripts/core/realscan/sources.py` | ROS RealSense + PiPER 수신 / 데모 / 가상 보정 소스 |
| `scripts/core/reconstruction/` | 물체 영역 분리 → ICP(선택) → TSDF → 메시 — UR/Isaac 스캔과 같은 코드 |
| `workcell/scans/piper_d405.yaml` | CAN·프레임·토픽·ROI 초기값·촬영/보정 허용 조건 |
| `workcell/robot/piper/piper_description.urdf` | agx_arm_ros 의 PiPER URDF 사본. **바이트 단위로 고치지 않는다** — 보정 파일이 이 파일의 해시를 기록한다 |

## 최초 설치

현재 컴퓨터에서는 마운트 연결과 패키지 설치를 완료했다. 컨테이너를 다시 준비할 때만 아래를 실행한다. 마운트 준비는 기존 컨테이너의 이름을 보존한 채 복제본을 만들므로 먼저 드라이버·MoveIt·RViz·카메라를 모두 Ctrl+C로 종료한다.

```bash
cd /home/sungjin/Documents/lg_sgu_vision_v2
python3 docker/prepare_piper_scan_container.py
docker exec ros2-jazzy-piper bash /workspace/lg_sgu_vision_v2/docker/install_piper_scan.sh
```

원래 컨테이너와 커밋 이미지는 백업으로 남는다. 이름과 설정은 `docker/.piper-backups/<시간>/rollback.txt`와 `inspect.json`에서 확인한다. 되돌릴 때는 모든 앱을 종료하고 새 컨테이너를 정지한 뒤 다른 이름으로 보관하고, 원래 백업 컨테이너를 `ros2-jazzy-piper`로 이름 변경해 시작한다.

## 매번 실행: 호스트 터미널 2개

먼저 일반 PiPER 드라이버와 MoveIt/RViz를 종료한다. 스캔 앱은 SDK로 CAN 관절 피드백을 직접 읽으므로 일반 드라이버가 필요 없다. 앱에는 로봇 명령 클라이언트가 없으며 SDK 통신 객체의 송신도 차단한다. 교시모드 전환과 팔 이동은 사용자가 로봇 버튼과 손으로 한다. 받침·작업 공간을 확인하고 각 촬영에서는 손과 팔을 멈춘다.

터미널 1 — 카메라:

```bash
docker start ros2-jazzy-piper
docker exec -it -e ROS_DOMAIN_ID=42 ros2-jazzy-piper bash \
  /workspace/agx_arm_ws/start_realsense_d405.sh
```

터미널 2 — 스캔 앱:

```bash
docker exec -it -e ROS_DOMAIN_ID=42 ros2-jazzy-piper bash \
  /workspace/lg_sgu_vision_v2/docker/start_piper_scan.sh --object piper_test --port 8083
```

브라우저에서 **http://localhost:8083**를 연다. 위 명령어는 카메라와 앱 모두 ROS domain 42를 사용한다. CAN `can0`는 호스트에서 1 Mbps로 준비돼 있어야 한다. 재부팅 등으로 CAN 설정이 초기화되었으면 앱 실행 전에 호스트에서 다음을 실행한다:

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
ip -details link show can0
```

오늘 촬영한 데이터·ROI·바닥 제외 설정을 그대로 불러오려면 터미널 2에서 위의 새 run 명령 대신 다음을 실행한다:

```bash
docker exec -it -e ROS_DOMAIN_ID=42 ros2-jazzy-piper bash \
  /workspace/lg_sgu_vision_v2/docker/start_piper_scan.sh --port 8083 \
  --resume-scan /workspace/lg_sgu_vision_v2/data/piper_test/scan/20261014_022204_317189
```

같은 카메라 장착 상태라면 검증된 보정을 재사용한다. 다시 촬영하려면 앱에서 **새 촬영 시작 (기존 결과 유지)**를 누르면 된다. 종료할 때는 터미널 2(Viser)와 터미널 1(카메라)에서 각각 Ctrl+C를 누르고, 두 프로그램이 종료된 뒤 호스트에서 `docker stop ros2-jazzy-piper`를 실행한다. 프로그램 종료와 컨테이너 정지는 로봇 본체의 모터·전원 차단 명령이 아니다.

앱은 카메라 일련번호·CameraInfo·광학 좌표 간 TF와 6개 관절 피드백을 확인한다. 정지 이력 0.5초, 관절 변화 0.002 rad 이하, 피드백 최신성 0.5초, 이미지·관절 시간 차이 0.1초 이내에서 촬영한다. 조건을 만족하지 않으면 촬영을 막고 이유를 표시한다. 교시에서 측정한 관절값은 URDF 한계로 잘라내지 않는다.

## 1. 카메라 장착 보정

1. [ChArUco 보드](../assets/piper_charuco_20mm.svg)를 **배율 100%, 페이지 맞춤 없이** 출력한다. 20 mm 눈금과 사각형 한 변이 실제 20 mm인지 자로 확인한다. 보드는 6×5칸, 마커 15 mm, `DICT_4X4_50`이며 평평하고 움직이지 않게 고정한다.
2. 카메라는 PiPER 손목에 단단히 고정한다. 보드를 잘 볼 수 있는 서로 다른 위치·회전에서 팔을 멈추고 **보정 자세 기록**을 누른다. 최소 15개를 기록하며 두 방향 이상의 충분한 회전이 필요하다. 같은 방향으로 평행 이동만 하면 보정을 거부한다.
3. **15개 이상으로 보정 계산**을 누른다. **보드 겹침·depth 검증 3D 보기**에서 학습 보드(파랑)가 평균 보드(노랑)에 겹치는지 확인한다. 서로 떨어지거나 회전이 어긋나 있으면 촬영·장착 상태를 점검한다.
4. 학습에 쓰지 않은 **새 자세 최소 3개**를 **별도 RGB·depth 검증 자세 기록**으로 촬영한다. 보드는 아직 제거하지 않는다. 검증 보드는 초록으로 겹쳐 표시하며, 마지막 검증의 주황 평면은 RGB에서 검출한 보드, 초록 점은 depth, 빨강 점은 거리 오차 기준을 초과한 depth다.
5. **검증 후 보정 저장**을 누른다. 보드 검출 재투영 오차는 2 px 이하, 학습·별도 검증의 고정 보드 편차는 현재 실험 설정으로 최대 5 mm·2° 이내여야 한다. RGB·depth 검사도 최소 3개 자세에서 통과해야 저장한다. 이 수치는 프로토타입의 허용 기준이며 검사 정밀도 보증값은 아니다.

학습·별도 검증 허용 편차는 `workcell/scans/piper_d405.yaml`의 `calibration_validation`에서 설정하며, 화면과 저장된 보정 보고서의 `limits`에 기록한다. 회전 허용치를 2°로 넓혀도 RGB·depth의 거리·각도 기준은 별도로 유지된다. 보드 자세의 일관성이 통과한 것이므로 실제 물체 복원 정확도는 별도로 확인한다.

RGB·depth 검사는 검출한 보드를 depth 영상으로 투영하고 가장자리 12 mm를 제외한 영역을 검사한다. 오차가 큰 점을 먼저 제거하지 않고, 유효 depth 비율 최소 35%·100점 이상, 보드 평면 거리 중앙값의 절댓값 3 mm 이하·절대 거리 오차 P95 5 mm 이하·평면 각도 차이 3° 이하를 확인한다. P95는 검사점의 95%가 해당 거리 오차 이내라는 뜻이다. RGB와 depth의 시간 차이·촬영 사이의 관절 변화도 검사한다.

실패한 검증은 저장되지만 성공한 검증 자세 개수에는 포함하지 않는다. 원인을 수정한 뒤 다시 기록한다. 보정 원본 폴더의 `depth_checks/*.npz`에는 RGB·depth 원본과 변환·잔차를, `depth_checks.json`에는 통계와 실패 이유를 저장한다. 허용 기준은 `workcell/scans/piper_d405.yaml`의 `depth_validation`에서 확인한다.

평면 검사는 깊이·평면 방향의 일치를 확인한다. 평면을 따라 움직이는 방향의 오차를 모두 식별할 수 없으므로 여러 자세의 보드 겹침 검사와 함께 사용한다. RGB·depth 검증이 없는 이전 실제 보정 파일은 새 보정이 필요하다는 메시지를 표시한다.

보정값은 `workcell/calibration/piper_d405.yaml`, 이미지·관절·보드 포즈 원본은 `workcell/calibration/samples/<run>/`에 저장한다. 실제 계산은 `base_T_tool(q) @ tool_T_color @ color_T_board`로 고정 보드 위치를 맞춘다. 복원에는 카메라 TF를 적용한 `tool_T_depth`를 사용한다. `base_link`와 `link6`은 PiPER URDF의 프레임이다.

카메라 장착을 바꾸면 반드시 다시 보정한다. 앱은 일련번호·해상도·내부 파라미터·TF·URDF 변경을 감지하지만, 같은 카메라의 물리적 장착 변경은 자동으로 알 수 없다. 보정 출력 패턴은 컨테이너 OpenCV와 맞추며 다른 ChArUco 패턴으로 대체하지 않는다.

### 계산 실패와 기록 복구

계산이 실패하면 버튼 아래에 이유와 원본 기록 수를 표시한다. 오차 기준을 넘은 계산 결과는 **미검증 진단용**으로 3D 보드와 자세별 회전 편차를 표시하며, 실제 보정값으로 저장하거나 촬영에 사용하지 않는다.

보드·카메라 고정 상태를 확인하고 품질이 낮은 자세를 다시 촬영하려면 **제외/복구할 자세 번호**를 입력하고 **선택 자세 제외/복구 (원본 유지)**를 누른다. 자동으로 기록을 제외하지 않는다. 제외한 기록은 회색으로 표시하고 원본 NPZ는 유지한다. 포함된 기록이 15개 미만이면 새 자세를 추가해야 계산할 수 있다. 제외·추가·재계산 후에는 별도 검증을 새로 진행한다.

앱을 종료한 뒤 저장한 보정 기록을 복구하려면 다음처럼 실행한다. 카메라 장착·보드 고정 상태는 원래 촬영과 같아야 한다.

```bash
docker exec -it ros2-jazzy-piper bash \
  /workspace/lg_sgu_vision_v2/docker/start_piper_scan.sh --port 8083 \
  --resume-calibration /workspace/lg_sgu_vision_v2/workcell/calibration/samples/실제_기록폴더
```

원본 이미지·기록과 현재 카메라·URDF·보드 규격의 일치를 확인하고 기록을 불러와 다시 계산한다. 새로운 별도 검증 파일은 이전 파일을 덮어쓰지 않는다. 이미 물체를 촬영한 run은 복구 옵션으로 이어 쓰지 않는다.

## 2. 물체 촬영

보드를 제거하고 5–15 cm 정도의 불투명 물체를 놓는다. 첫 촬영에서는 약 20 cm 거리에서 물체 전체가 보이게 한다. 설정의 유효 depth 범위는 7–50 cm다.

**화면 중앙 물체에 ROI 놓기**로 대략적인 위치를 잡고 ROI 중심·반경 또는 3D 핸들을 조정한다. 좌표 단위는 PiPER `base_link` 기준 m다. 테이블을 제거하려면 **테이블 제외**를 켜고 **바닥 제외 높이 z [m]**를 설정한다. 먼저 실제 테이블 높이로 시작하고 깊이 잡음으로 바닥이 남으면 조금씩 올려 다시 복원한다. 이 값은 제거 영역의 윗면이며, 너무 올리면 물체 하단도 잘린다. **추가 영역 제외 (받침대·잡음)**로 물체와 떨어진 받침대·잔여 조각을 별도 상자로 제외할 수 있다. 잘못된 높이·상자는 물체까지 제거하므로 먼저 점군으로 확인한다.

복원 결과를 불러오거나 새 복원이 끝나면 현재 카메라 점군과 촬영 점군을 기본으로 숨겨 메시를 확인하기 쉽게 한다. **현재 카메라 점군 보기**와 **촬영 점군 보기**는 각각 켤 수 있다. 현재 카메라 점군은 원본 depth 미리보기이므로 테이블을 포함할 수 있으며, 최종 복원 메시와 구분한다.

**현재 카메라 위치·촬영 방향 보기**는 검증된 장착 보정과 측정 관절값으로 계산한 depth 카메라 위치를 주황색 시야 모양과 방향선으로 표시한다. 교시로 이동하는 동안에도 약 0.5초마다 갱신하며, 저장된 촬영 시점은 파란색으로 구분한다. 미리보기 패널에 `base_link` 기준 x·y·z와 ROI 중심까지의 직선 거리를 표시한다. 주황색 선은 카메라가 바라보는 방향이며 길이는 표시용이다. **카메라 아이콘**에서 모양을 고른다 — 작은 시야(기본, 선 8 cm) / 점 + 방향선(8 cm) / 큰 시야(이전 모양, 선 20 cm). 저장된 촬영 시점(파랑)의 크기도 함께 바뀐다. 점군을 숨겨도 카메라 표시는 유지된다. 보정이 없거나 관절·카메라 피드백이 오래되면 현재 위치 표시를 숨기고 이유를 표시한다. 위치 미리보기는 이동 중에도 가능하지만, 실제 촬영의 정지 확인 조건은 유지한다.

겹치는 면이 보이도록 교시로 자세를 바꾼 뒤 정지하고 **정지한 자세 촬영**을 누른다. 권장은 8–12개 시점이다. 현재 프레임의 depth와 해당 시간의 측정 관절값으로 카메라 포즈를 계산하며, 앱은 팔을 움직이지 않는다.

처음부터 다시 찍으려면 **새 촬영 시작 (기존 결과 유지)**를 누른다. 새 시간 이름의 run을 만들고 촬영 개수를 0장으로 바꾸며, 기존 사진·복원 파일은 기존 폴더에 그대로 둔다. 검증된 보정값과 현재 ROI·바닥/추가 제외 영역 설정을 유지한다. 다른 물체라면 먼저 **물체 이름**을 바꾸고 새 촬영을 시작한다. 물체 위치·받침 높이가 달라졌다면 ROI와 제외 영역을 다시 조정하며, 카메라 장착이 달라졌다면 새 보정이 필요하다. 기존 촬영에 시점만 추가하려면 새 촬영 버튼 없이 계속 촬영한다.

## 3. 복원과 결과

먼저 voxel 1 mm, ICP 꺼짐으로 **촬영 프레임으로 복원**을 실행한다. 여러 시점의 표면이 어긋나면 보정을 먼저 확인한 뒤 필요할 때 ICP를 시도한다. 보지 못한 바닥면·가려진 표면은 복원되지 않을 수 있다.

결과 경로는 `data/<object>/scan/<run>/`이다. 같은 run을 덮어쓰지 않으며 기본 run 이름은 실행 시간이다.

같은 물체·로봇 베이스·카메라 장착 상태에서 앱을 재실행하고 촬영을 이어가려면 **촬영 복구** 옵션을 사용한다:

```bash
docker exec -it ros2-jazzy-piper bash \
  /workspace/lg_sgu_vision_v2/docker/start_piper_scan.sh --port 8083 \
  --resume-scan /workspace/lg_sgu_vision_v2/data/piper_test/scan/실제_run_이름
```

현재 보정 파일이 촬영 당시 보정값과 일치하는지, 촬영 기록과 프레임 번호가 맞는지 확인한다. 기존 원본·ROI·테이블/받침대 설정·촬영 개수를 복구하고 다음 번호로 저장하므로 기존 프레임을 덮어쓰지 않는다. 보정 복구 옵션과는 함께 사용하지 않는다. 물체나 카메라 장착이 바뀌었다면 새 run으로 촬영한다.

| 파일 | 내용 |
|---|---|
| `frames/*.npz` | depth[m], K, `world_T_cam`, 측정 관절값 |
| `capture.json` | 촬영 시간·시간 차이·카메라·보정 기록 |
| `plan.npz`, `plan.json` | ROI·테이블/받침대·카메라 설정 스냅샷 |
| `recon.ply`, `recon.obj`, `recon.json` | base_link 좌표의 복원 메시·통계 |
| `recon_poses.npz` | 복원에 사용한 카메라 포즈 |

ICP를 켜면 `recon_icp.*`로 별도 저장한다. 기존 재구성 저장 형식을 사용하지만 이 데이터를 곧바로 UR 검사 궤적 실행에 넣지는 않는다.

저장 결과만 보기 — 로봇과 카메라에 연결하지 않는다:

```bash
docker exec -it ros2-jazzy-piper bash \
  /workspace/lg_sgu_vision_v2/docker/start_piper_scan.sh \
  --view-run /workspace/lg_sgu_vision_v2/data/piper_test/scan/실제_run_이름
```

하드웨어 없이 합성 구 데이터로 확인:

```bash
docker exec -it ros2-jazzy-piper bash \
  /workspace/lg_sgu_vision_v2/docker/start_piper_scan.sh --demo --demo-frames 8
```

데모는 `data/piper_demo/scan/`에 저장되며 실제 보정에 사용할 수 없다.

## 실제 보정 전 가상 테스트

보드 영상의 ChArUco 검출부터 손목 카메라 보정·별도 검증·물체 촬영·TSDF 복원까지 하드웨어 없이 시험할 수 있다. 이 모드는 카메라 투영과 좌표 변환의 **기하 시뮬레이션**이며, PiPER 관절 운동학·도달 가능성·충돌·물리·실제 센서 잡음을 재현하지 않는다. 로봇 드라이버와 카메라를 켤 필요가 없다.

자동으로 15개 보정 자세, 3개 별도 검증 자세, 8개 구형 물체 시점을 처리하고 복원 화면을 띄운다:

```bash
docker exec -it ros2-jazzy-piper bash \
  /workspace/lg_sgu_vision_v2/docker/start_piper_scan.sh \
  --sim-calibration --sim-auto
```

브라우저에서 http://localhost:8082를 연다. 보정 패널에는 알려진 가상 장착값 대비 위치·회전 오차, 복원 패널에는 메시 크기와 결과 경로가 표시된다. 촬영된 점군과 카메라 시점을 3D로 살펴볼 수 있다.

버튼을 직접 눌러 순서를 시험하려면 `--sim-auto`를 빼고 실행한다. **보정 자세 기록**을 15번 누르면 각 기록마다 다음 가상 시점으로 이동한다. **보정 계산 → 별도 RGB·depth 검증 자세 기록 3번 → 검증 후 보정 저장** 순서로 진행하면 보드가 물체로 바뀐다. 이후 **정지한 자세 촬영**을 8번 누르고 복원을 실행한다.

깊이 오차를 넣어 실패 동작을 확인하려면 다른 포트에서 실행한다:

```bash
docker exec -it ros2-jazzy-piper bash \
  /workspace/lg_sgu_vision_v2/docker/start_piper_scan.sh \
  --sim-calibration --sim-auto --sim-depth-bias-mm 10 --port 8084
```

이 경우 보드의 학습 겹침은 정상이지만 depth 점군이 평면에서 떨어져 빨강으로 표시된다. 자동 검증은 중단되고 화면은 유지된다. **가상 depth 거리 오차 [mm]**를 0으로 바꾼 후 **별도 RGB·depth 검증 자세 기록**을 3번 누르고 저장하면 정상 흐름으로 진행할 수 있다. 검증 실패 상태에서는 물체 촬영을 허용하지 않는다.

가상 보정은 `workcell/calibration/simulation/`에, 물체 데이터는 `data/piper_sim/scan/<run>/`에 저장한다. `demo: true`로 표시되어 실제 보정으로 로드되지 않는다. 실제 장착 오차와 센서 오차는 현장 보정으로 별도 확인해야 한다.

SSH로 원격 접속한 경우, **원격으로 접속하는 PC**에서 포트 전달을 연다:

```bash
ssh -N -L 8082:127.0.0.1:8082 sungjin@로봇PC주소
```

그 PC의 브라우저에서 http://localhost:8082를 연다. 원격 데스크톱 안에서 로봇 PC의 브라우저를 쓰는 경우에는 포트 전달이 필요 없다.
오차 화면도 함께 보려면 같은 SSH 명령에 `-L 8084:127.0.0.1:8084`를 추가한다.

## 종료·문제 해결

두 터미널에서 각각 Ctrl+C로 종료한다. 스캔 종료는 모터 전원·교시모드를 바꾸지 않는다. 이후 MoveIt으로 돌아갈 때 기존 PiPER 교시 복구 절차를 따르며, **스캔 촬영을 위해 CAN 제어 복구/이동 명령을 실행할 필요는 없다.**

- 관절 피드백 대기: 전원·CAN 연결·호스트 `can0` 상태를 확인한다. 앱은 피드백 복구를 위해 모터를 켜지 않는다.
- 기존 드라이버/MoveIt 실행 중: 해당 터미널을 종료한 후 스캔 앱을 다시 실행한다.
- 카메라 대기: 카메라 터미널과 두 프로세스의 ROS domain, `workcell/scans/piper_d405.yaml` 토픽 이름을 확인한다.
- 보정/검증 실패: 보드 실제 치수·고정 상태·카메라 흔들림·다양한 회전을 확인하고 다시 촬영한다.
- 포트 사용 중: 기존 스캔 앱을 종료하거나 `--port 8083`을 지정한다.

하드웨어 명령 없이 실행하는 회귀 검증:

```bash
docker exec ros2-jazzy-piper bash -lc '
  source /opt/ros/jazzy/setup.bash
  /workspace/lg_sgu_vision_v2/.venv-piper/bin/python \
    /workspace/lg_sgu_vision_v2/scripts/checks/check_piper_scan.py
'
```

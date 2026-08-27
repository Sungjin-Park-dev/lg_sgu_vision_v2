# LooksGood 핸드북

물체 표면의 검사 지점을 찍는 것부터, 로봇이 그 지점들을 실제로 훑고 돌아오기까지 — 전 과정을 처음 쓰는 사람 기준으로 정리했다.

> 💡 **여기에 노션 목차 블록을 넣는다**
>
> 이 줄에서 `/목차` (영문 `/toc`) 를 입력하면 아래 헤딩으로 목차가 자동 생성된다. 넣은 뒤 이 안내는 지운다.

---

## LooksGood이란

LooksGood은 **검사할 물체를 놓고, 어디를 볼지 정하고, 로봇이 거기를 훑는 경로를 만들어, 시뮬레이션과 실제 로봇에서 실행**하는 한 벌의 도구다. 세 개의 앱으로 나뉘어 있고, 앞 앱의 출력이 다음 앱의 입력이 된다.

| 단계 | 앱 | 하는 일 | 결과물 |
| --- | --- | --- | --- |
| 1 | `viewpoint_studio.py` | 물체 표면을 카메라 화각으로 덮는 검사 지점(viewpoint) 생성 | `.h5` |
| 2 | `isaac_pipeline.py` | IK 풀이 · 충돌 회피 경로 생성 · 미리보기 · 실행 | `.csv` |
| 보조 | `trajectory_studio.py` | Isaac 없이 브라우저에서 경로를 계획·확인 (선택) | `.csv` |

### 데이터가 흘러가는 순서

**파일 흐름**

```
data/{물체}/mesh/source.obj                              ← 준비물
      ↓
data/{물체}/viewpoint/{N}/viewpoints.h5           ← Viewpoint Studio
      ↓
data/{물체}/ik/{N}/ik_*.h5                               ← Solve IK
      ↓
data/{물체}/trajectory/{N}/solution.h5                   ← 방문 순서
      ↓
data/{물체}/trajectory/{N}/trajectory.csv (+ .npz)       ← 실행 경로
      ↓
Isaac 미리보기  또는  실제 로봇
```

`{N}`은 그 물체의 viewpoint 개수다. 같은 물체라도 viewpoint 수가 다르면 다른 폴더에 저장되므로, 여러 설정을 나란히 두고 비교할 수 있다.

> 📷 **PLATE 01 · 사진**
>
> 세 앱의 화면을 나란히 놓은 한 장. 왼쪽부터 Viewpoint Studio(브라우저), Isaac Pipeline(Isaac Sim 창), RViz. 전체 그림을 먼저 보여주는 자리다.

---

## 환경 셋팅

모든 작업은 `ros-jazzy` 컨테이너 하나 안에서 돌아간다. 호스트에는 Docker와 NVIDIA 드라이버만 있으면 된다.

### 사전 준비물

| 항목 | 요구사항 | 확인 방법 |
| --- | --- | --- |
| GPU | NVIDIA, VRAM 16 GB 이상 권장 | `nvidia-smi` |
| 드라이버 | 580 이상 (cuMotion이 CUDA 13을 쓴다) | `nvidia-smi` 우측 상단 |
| Docker | NVIDIA Container Toolkit 설치됨 | `docker info \| grep -i nvidia` |
| 디스플레이 | Isaac Sim 창을 띄울 X 서버 | `echo $DISPLAY` |

> ⚠️ **주의 · 드라이버 버전**
>
> 드라이버가 **580 미만**이면 Isaac Pipeline은 뜨지만 경로 생성에서 `CUDA driver version is insufficient for CUDA runtime version`으로 죽는다. 컨테이너가 CUDA 13을 싣고 있는데 드라이버가 그보다 낮아서 생기는 문제다. 드라이버를 올리는 것 말고 우회 방법은 없다.

### 설치 — 최초 1회

프로젝트 루트에서 아래 네 줄을 순서대로 실행한다.

**호스트 · 프로젝트 루트**

```bash
# 1) 컨테이너가 호스트 화면을 쓸 수 있게 허용 (Isaac Sim 창)
xhost +local:root

# 2) 이미지 빌드 + 컨테이너 시작
docker compose -f docker/compose.yaml up -d --build

# 3) 파이썬 venv · ROS overlay · Julia depot 설치
docker exec -it ros-jazzy bash /workspace/docker/install_env.sh

# 4) 설치가 제대로 됐는지 점검
docker exec -it ros-jazzy bash /workspace/docker/verify_env.sh
```

2번은 처음 실행할 때 이미지를 내려받고 빌드하느라 오래 걸린다. 3번도 마찬가지다. 4번이 모두 통과하면 준비가 끝난 것이다.

> 📷 **PLATE 02 · 사진**
>
> `verify_env.sh`가 전부 통과했을 때의 터미널 출력. 무엇이 통과로 떠야 정상인지 보여주는 기준 화면.

### 매번 하는 일 — 컨테이너 들어가기

**호스트 · 앱을 켤 때마다**

```bash
xhost +local:root            # 재부팅했다면 다시
docker start ros-jazzy       # 꺼져 있다면
docker exec -it ros-jazzy bash
```

> 💡 **셸을 여러 개 쓴다**
>
> MoveIt으로 제어할 때는 터미널이 **두 개** 필요하다. 둘 다 위 `docker exec -it ros-jazzy bash`로 같은 컨테이너에 들어간 것이면 된다. 새 탭을 열어 같은 명령을 다시 치면 된다.

### 앱 실행 명령

**컨테이너 · /workspace**

```bash
# 뷰포인트 생성  →  브라우저 http://localhost:8080
uv run scripts/apps/viewpoint_studio.py

# 경로 계획(선택) →  브라우저 http://localhost:8081
uv run scripts/apps/trajectory_studio.py

# Isaac Pipeline  →  Isaac Sim 창이 뜬다
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync \
  scripts/apps/isaac_pipeline.py --object sample
```

---

## Viewpoint 생성

Viewpoint는 **카메라가 서야 할 자리와 바라볼 방향**이다. 물체 표면 위 한 점에서 법선 방향으로 working distance만큼 띄운 위치에 카메라를 놓는다. 이 지점들을 빠짐없이 찍는 것이 검사다.

이 앱이 만드는 것은 **지점들과 그 지점들을 잇는 그래프** 둘뿐이다. **어느 순서로 돌지는 여기서 정하지 않는다** — 방문 순서와 팔 자세는 다음 단계(Isaac Pipeline)가 함께 푼다.

### 실행 방법

**컨테이너 · Viewpoint Studio**

```bash
uv run scripts/apps/viewpoint_studio.py
```

브라우저에서 `http://localhost:8080`에 접속한다. 3D 뷰가 왼쪽 대부분을 차지하고, 오른쪽에 접이식 설정 패널이 붙는다.

> 📷 **PLATE 03 · 사진**
>
> Viewpoint Studio 전체 화면. 3D 뷰에 물체 메시·표면점·카메라 위치·그래프 간선이 모두 켜져 있고, 오른쪽 패널이 펼쳐진 상태. 하단 진단 한 줄이 함께 보이면 좋다.

### 화면 구성

| 영역 | 역할 |
| --- | --- |
| `Object` | 물체 선택. 패널 맨 위에 접히지 않고 항상 보인다. **바꾸면 화면이 비워진다** |
| `Viewpoints` | 이 물체의 viewpoint를 어디서 얻을지 — 저장본을 열거나, 새로 만든다 |
| `Display` | 화면에 무엇을 그릴지 켜고 끄는 토글만 모여 있다 |
| `Solver graph (hops)` | 다음 단계가 실제로 풀 그래프로 관점을 바꾸는 슬라이더 |
| 진단 한 줄 | 패널 맨 아래. 지금 그래프의 상태를 숫자로 말한다 |

### 물체 및 Viewpoint 불러오기

| 컨트롤 | 의미 |
| --- | --- |
| `Object` | `data/` 아래에 메시가 있는 물체 목록. 고르면 3D 뷰에 그 물체가 뜬다. 목록에 없다면 메시부터 준비해야 한다 |
| `Saved viewpoints` | 저장된 파일을 다시 연다. 열면 **그 파일에 저장된 카메라 스펙이 입력칸에 그대로 채워진다** — 어떤 설정으로 만든 것인지 되짚을 수 있다 |

---

| 표시 | 뜻 |
| --- | --- |
| `(none)` | 아직 아무것도 없다 |
| `(generated · unsaved)` | 방금 만들었지만 **아직 저장하지 않았다** |
| 파일 이름 | 그 파일을 불러온 상태 |

### 카메라 스펙

세 값이 viewpoint의 위치를 직접 결정한다. 그리고 이 값들은 `.h5`에 함께 저장되어 **이후 모든 단계(IK · 경로 생성 · Isaac)가 이 값을 쓴다.** 나중에 바꾸려면 viewpoint를 다시 만들어야 한다.

| 컨트롤 | 의미 | 영향 |
| --- | --- | --- |
| `FOV width (mm)` | working distance에서 카메라가 담는 가로 실폭 | 클수록 viewpoint 수가 준다 |
| `FOV height (mm)` | 같은 거리에서의 세로 실폭 | 〃 |
| `Working distance (mm)` | 표면에서 카메라까지 띄우는 거리 | **로봇이 실제로 가는 위치가 바뀐다** |

> 💡 **Working distance에는 하한이 있다**
>
> 렌즈 물리 제약이라, 그보다 작은 값을 넣으면 생성이 거부된다. 검사면이 렌즈 배럴 안쪽에 놓이는 값이기 때문이다.

### 생성 파라미터

---

| 컨트롤 | 범위 | 무엇을 정하나 |
| --- | --- | --- |
| `FOV overlap (%)` | — | 이웃한 촬영 영역이 겹치는 비율. **점의 개수와 간격**을 정한다. 표면 간격 = min(FOV) × (1 − overlap). 올리면 촘촘해지고 검사 시간이 늘어난다 |
| `Max edge length (×)` | 1.0 – 5.0 | 간선 길이 상한. **주변 카메라 위치 간격의 배수**다 (표면 간격이 아니다) |
| `Max normal angle (°)` | 15 – 180 | 두 점의 법선이 이보다 벌어지면 잇지 않는다. `90°`면 반대편 면이 차단된다 |
| `Neighbor search (k)` | 3 – 30 | 삼각분할 후보로 볼 이웃 수. 거의 건드릴 일이 없다 |

> 💡 **뒤의 셋이 그래프를 만든다**
>
> `Max edge length` · `Max normal angle` · `Neighbor search` 셋이 **Delaunay 인접 그래프**를 만든다. 이 그래프가 다음 단계의 순서 제약이 되므로, 간선이 어떻게 이어지는지가 로봇의 이동 비용을 좌우한다.
>
> 네 값 모두 **슬라이더가 아니라 숫자 입력칸**이다. 고쳐도 `Generate`를 누르기 전까지는 화면이 바뀌지 않는다.

#### 실행

| 버튼 | 동작 |
| --- | --- |
| `Generate` | 위 설정으로 viewpoint를 만든다. 끝나면 바로 아래에 `Done · 132 vp · 329 edges · 2 component(s)` 처럼 결과가 뜬다 |
| `Save h5` | 만든 결과를 파일로 저장한다. **누르지 않으면 남지 않는다** — 저장 전에는 `Saved viewpoints`가 `(generated · unsaved)`로 표시된다 |

> 📷 **PLATE 04 · 사진**
>
> `Viewpoints` 폴더를 펼친 확대 컷. `Saved viewpoints` → `Camera spec` → `Generate viewpoints` 순서와 노브 네 개, 그 아래 `Done ·` 상태 줄이 한눈에 들어오게. 툴팁 하나를 띄운 상태면 더 좋다.

### 결과 확인

---

| 토글 | 기본 | 보여주는 것 |
| --- | --- | --- |
| `Mesh` | 켬 | 물체 표면 |
| `Surface points` | 켬 | 메시 표면 위의 검사 지점 |
| `Camera positions` | 켬 | 표면점 + 법선 × WD — **로봇 EE가 실제로 가는 곳** |
| `Graph edges` | 켬 | 다음 단계의 순서 제약 그래프. **색이 연결 성분**을 나타낸다 |

---

> 💡 **hops는 다음 단계와 맞춰 둔다**
>
> 파일에 저장되는 간선은 항상 1-hop이고, 경로 생성이 그것을 N-hop으로 확장해 푼다. 이 슬라이더를 Isaac Pipeline의 `--delaunay-expand-hops`와 **같은 값**으로 두면 화면과 다음 단계가 같은 그래프를 본다. 양쪽 기본은 `2`다.

#### 진단 한 줄 읽기

**화면 · 패널 맨 아래**

```
329 edges · 2 components · 0 isolated · GLNS: 852 (2-hop)
```

| 항목 | 뜻 |
| --- | --- |
| `edges` | 저장될 간선 수 (1-hop) |
| `components` | 그래프가 몇 조각인지. **2 이상이어도 대개 정상이다** — 조각 사이를 잇는 이동 구간이 생길 뿐이고, 실제로 쓰는 파일 다수가 2성분 이상이다 |
| `isolated` | 간선이 하나도 없는 외톨이 점. 다음 단계가 제약 그래프에서 떨어뜨린다 |
| `GLNS: N (h-hop)` | 현재 hops 설정에서 다음 단계가 실제로 푸는 간선 수 |

> ⚠️ **주의 · No graph**
>
> 진단 줄에 `⚠ No graph`가 뜨면 그 파일에 Delaunay 간선이 없다는 뜻이다. 경로 생성이 거부하므로 다시 만들어야 한다.

> 📷 **PLATE 05 · 사진**
>
> 같은 물체를 `Max normal angle`만 바꿔 두 번 생성한 비교 컷. 성분이 여러 조각으로 갈라진 경우와 하나로 이어진 경우를, 하단 진단 한 줄과 함께 나란히.

### Viewpoint 저장

---

**저장 위치**

```
data/{물체}/viewpoint/{개수}/viewpoints.h5

# 예
data/cylinder_sample/viewpoint/132/viewpoints.h5
```

이 파일 경로를 다음 단계인 Isaac Pipeline에 그대로 넣는다. 파일 안에는 각 지점의 위치·법선·카메라 위치와 함께 **카메라 스펙, 그래프 간선**이 들어 있다. 방문 순서는 들어 있지 않다.

### 작업 순서 요약

1. `Object`에서 물체를 고른다.
2. `Camera spec`에서 FOV와 working distance를 정한다.
3. `Generate viewpoints`에서 overlap과 그래프 노브 3개를 조절한다.
4. `Generate`로 만든다.
5. `Display` 토글과 하단 진단 한 줄로 점 분포와 그래프를 확인한다.
6. 마음에 들면 `Save h5`로 저장한다.

---

## 모드 설명

Isaac Pipeline은 두 개의 축으로 동작을 정한다. **어느 로봇을 움직일 것인가**(Run 모드)와 **누가 명령을 내리는가**(Pipeline 모드)다. 두 축이 각각 두 값이라 조합은 네 가지다.

|  | Simulation | Real |
| --- | --- | --- |
| MoveIt! | **Sim 로봇을 MoveIt!으로 제어** — RViz에서 목표를 찍고 Plan & Execute하면 Isaac 안의 UR20이 움직인다. | **Real 로봇을 MoveIt!으로 제어** — RViz 명령이 실제(또는 mock) 로봇으로 나간다. Isaac의 로봇은 실물을 따라 움직이는 거울이다. |
| Inspection | **Sim 환경에서 검사 수행** — 앱이 만든 검사 경로를 Isaac 안의 UR20에서 실행한다. ROS 없이도 된다. | **Real 로봇으로 검사 경로 실행** — 같은 경로를 실제 로봇 컨트롤러로 보낸다. Isaac의 로봇은 실물을 따라 움직이는 거울이다. |

### 모드 고르는 법

모드는 앱을 켤 때 옵션으로 주고, 켠 뒤에는 앱 맨 위 두 콤보박스에서 바꿀 수 있다.

| 축 | 실행 옵션 | UI 위치 | 선택지 |
| --- | --- | --- | --- |
| 명령 주체 | `--pipeline-mode` | `Pipeline Mode` | `inspection` · `moveit` |
| 대상 로봇 | `--mode` | `Run Mode` | `sim` · `real` |

> ⚠️ **주의 · 실행 중에는 바꾸지 않는다**
>
> 경로 생성이나 실행이 도는 동안에는 두 콤보박스가 **자동으로 잠긴다.** 도중에 모드를 바꾸면 이미 목표를 정한 작업의 발밑이 바뀌기 때문이다. 작업이 끝나면 다시 풀린다.

> 🤖 **실로봇 · ROS 스택과 짝이 맞아야 한다**
>
> Run 모드가 `real`이면 ROS 쪽도 real 스택을 띄워야 하고, `sim`이면 sim 스택을 띄워야 한다. **두 스택을 동시에 띄우면 안 된다** — 로봇 모델이 충돌해 ROS 노드가 죽는다.

> 📷 **PLATE 06 · 사진**
>
> Isaac Pipeline 상단의 `Pipeline Mode`와 `Run Mode` 콤보박스 확대 컷. 네 조합 중 어디에 있는지가 화면에서 어떻게 보이는지.

---

## 제어 모드 — MoveIt!

로봇을 **임의의 자세로 움직이고 싶을 때** 쓴다. 검사 경로와 무관하게 팔을 옮기거나, 자세를 잡거나, 충돌 없이 어딘가로 보내는 일이다. 명령은 RViz에서 내린다.

### MoveIt × Simulation

Isaac 안의 로봇을 RViz로 움직인다. **터미널 두 개**가 필요하고, 순서가 중요하다.

#### 셸 1 — Isaac 앱 (먼저)

**SIM · 셸 1**

```bash
source /workspace/.venv/bin/activate
OMNI_KIT_ACCEPT_EULA=YES python scripts/apps/isaac_pipeline.py \
  --object sample --mode sim --pipeline-mode moveit
```

> ⚠️ **주의 · 창이 뜨면 ▶ Play를 누른다**
>
> Isaac Sim 뷰포트 왼쪽의 **▶ Play** 버튼을 눌러야 로봇 상태가 ROS로 흘러나간다. 이걸 안 누르고 셸 2를 실행하면 컨트롤러가 로봇을 못 찾고 `Switch controller timed out`으로 실패한다.

`uv run` 대신 `source .venv/bin/activate` 후 `python`으로 실행하는 이유는, MoveIt이 쓰는 ROS 브리지가 그렇게 해야 제대로 잡히기 때문이다.

#### 셸 2 — ROS 스택 (Play를 누른 뒤)

**SIM · 셸 2**

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_overlay/install/setup.bash
ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py
```

컨트롤러가 붙고 계획기와 RViz가 뜨면 터미널에 `You can start planning now!`가 출력된다. 이때부터 RViz를 쓰면 된다.

### MoveIt × Real

셸 1의 Isaac 앱을 `--mode real --pipeline-mode moveit`으로 켜고, 셸 2에서 real 스택을 띄운다. **기본값은 실제 로봇이 아니라 mock hardware**라, 로봇 없이도 전체 흐름을 그대로 연습할 수 있다.

**REAL · 셸 2 · mock hardware (기본)**

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_overlay/install/setup.bash
ros2 launch scripts/moveit/ur20_real_moveit.launch.py
```

| launch 인자 | 기본값 | 의미 |
| --- | --- | --- |
| `use_mock_hardware` | true | `true`면 로봇 없이 가짜 하드웨어로 돈다. 실제 로봇을 쓰려면 `false` |
| `robot_ip` | 0.0.0.0 | 실제 로봇의 IP. mock일 때는 무시된다 |
| `ur_type` | ur20 | 로봇 기종 |
| `scene` | sim_default | 장애물 배치가 적힌 워크셀 정의 이름 |
| `use_cumotion` | true | `false`로 두면 GPU 계획기 없이 계획한다. 드라이버 문제로 cuMotion이 안 뜰 때 임시 우회 |

> 🤖 **실로봇 · 붙이기 전에**
>
> 이 문서는 실제 로봇을 곧바로 움직이는 복사·실행용 명령을 싣지 않는다. 현장 네트워크와 안전 설정을 확인한 뒤, `use_mock_hardware:=false`와 로봇 IP를 직접 넣어 실행한다.
>
> 드라이버를 띄운 뒤 **펜던트에서 외부 제어 프로그램을 실행**해야 ROS 명령이 로봇에 전달된다. launch 하나가 드라이버와 MoveIt을 같이 띄우므로, 그 사이에 펜던트를 조작하면 된다.

### RViz 사용법

1. RViz의 상호작용 마커(로봇 손끝의 화살표·링)를 끌어 목표 자세를 만든다.
2. **Plan**을 눌러 경로를 계산한다. 계산된 경로가 반투명 로봇으로 재생된다.
3. 경로가 마음에 들면 **Execute**를 누른다. 그때 비로소 로봇이 움직인다.
4. 한 번에 하려면 **Plan & Execute**를 누른다.

평소에는 MoveIt이 로봇의 현재 자세를 그대로 반영하고 있다가, Execute를 누르는 순간에만 움직인다.

> 📷 **PLATE 07 · 영상**
>
> RViz에서 마커를 끌어 목표를 잡고 → Plan으로 경로를 미리 보고 → Execute로 Isaac 로봇이 움직이는 20초 클립. 왼쪽 RViz, 오른쪽 Isaac Sim을 한 화면에 담으면 좋다.

> ⚠️ **주의 · 자주 걸리는 두 가지**
>
> **순서.** 셸 1을 켜고 ▶ Play를 누른 *다음에* 셸 2를 실행한다. 반대로 하면 컨트롤러가 타임아웃된다.
>
> **중복.** 셸 2 스택은 한 번에 하나만 띄운다. sim과 real 스택을 같이 띄우면 로봇 모델이 충돌해 노드가 죽는다.

---

## 검사 모드 — Inspection

LooksGood의 본체다. viewpoint 파일을 받아 **IK를 풀고, 충돌 없는 검사 경로를 만들고, 미리 보고, 실행**한다. 명령은 Isaac Pipeline 앱의 패널에서 내린다.

### 실행

**SIM · 가장 안전한 기본 조합 · 셸 하나로 충분**

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync \
  scripts/apps/isaac_pipeline.py --object sample \
  --mode sim --pipeline-mode inspection
```

이 조합은 ROS 스택 없이 돌아간다. 경로 생성부터 미리보기, Isaac 로봇 실행까지 이 창 하나에서 끝난다. 실제 로봇으로 넘어가기 전에 여기서 충분히 확인하는 것이 표준 순서다.

> 📷 **PLATE 08 · 사진**
>
> Isaac Pipeline 전체 화면. 오른쪽 패널이 위에서부터 순서대로 보이고, 뷰포트에는 로봇·테이블·물체·viewpoint 점이 함께 있는 상태.

### 패널 순서 = 작업 순서

패널은 위에서 아래로 일하는 순서대로 놓여 있다.

| # | 패널 | 하는 일 |
| --- | --- | --- |
| 1 | `Pipeline Mode` | 명령 주체 선택 (Inspection / MoveIt) |
| 2 | `Run Mode` | 대상 로봇 선택 (Simulation / Real) |
| 3 | `Load Object & Viewpoints` | 물체를 놓고 viewpoint 파일을 고른다 |
| 4 | `Solve IK` | 각 viewpoint에 팔이 닿는지 확인하고 해를 저장 |
| 5 | `Motion Speed` | 만들어질 경로의 실행 속도를 정한다 |
| 6 | `Generate Trajectory` | 검사 경로 또는 tilt 경로를 만든다 |
| 7 | `Preview in Simulation` | 고스트 로봇으로 재생해 확인 |
| 8 | `Execute Trajectory` | 실제로 움직인다 |
| — | `Scene (obstacles)` | 워크셀 실측 보조 도구 (평소엔 접어 둔다) |
| — | `Log` | 모든 작업의 출력이 흘러나오는 곳 |

### 3 · Load Object & Viewpoints

검사 대상을 정의하는 자리다. 물체가 **어디에 놓여 있는지**가 이후 모든 계획의 전제가 된다.

| 컨트롤 | 의미 |
| --- | --- |
| `Object` | 불러올 물체 선택 |
| `Load Object` | 고른 물체를 화면에 올리고, 아래 입력한 위치·자세를 적용한다. **배치 적용 버튼을 겸한다** |
| `Read Pose` | 반대 방향. 화면에서 기즈모로 옮긴 물체의 현재 위치를 입력칸으로 읽어 온다 |
| `Frame` | 아래 숫자를 어느 기준으로 읽을지. `base_link`(로봇 URDF 기준) 또는 `base`(UR 펜던트·ArUco 기준) |
| `Object pose (m)` / `Object rotvec` | 물체의 위치(m)와 회전(회전벡터) |
| `Viewpoints (h5)` | 2단계에서 저장한 파일. `Browse...`로 고른다 |
| `Show Viewpoints` / `Clear Viewpoints` | 그 파일의 검사 지점을 물체 위에 점으로 그리거나 지운다. **물체가 먼저 올라와 있어야 한다** |

#### ArUco 마커로 위치 잡기

---

| 컨트롤 | 의미 |
| --- | --- |
| `Marker pose (m)` / `Marker rotvec` | 측정한 마커의 위치·자세. 위 `Frame` 기준으로 읽힌다 |
| `Object in marker (m)` / `Object rotvec` | 마커에서 본 물체의 위치·자세. 마커 기준이라 `Frame`과 무관하다 |
| `Marker size (m)` | 마커 한 변의 길이. 화면에 그릴 때만 쓴다 |
| `Show Marker` / `Hide Marker` | 마커를 화면에 표시하거나 감춘다. 측정이 맞는지 눈으로 확인하는 용도 |

> 💡 **기즈모로 옮겨도 된다**
>
> 숫자를 넣는 대신 뷰포트에서 물체를 직접 끌어 옮겨도 된다. 경로 생성은 **화면에 보이는 현재 위치**를 그대로 쓴다. 옮긴 값을 기록해 두고 싶으면 `Read Pose`로 읽어 낸다.

> 📷 **PLATE 09 · 사진**
>
> `Show Viewpoints`를 누른 뒤의 뷰포트. 물체 표면 위에 검사 지점들이 점으로 찍혀 있는 모습.

### 4 · Solve IK

각 viewpoint에 대해 **팔이 그 자세를 만들 수 있는지**를 확인하고, 가능한 관절 조합(IK 해)을 파일로 저장한다. 물체·viewpoint당 한 번만 하면 되고, 이후 경로 생성이 이 결과를 재사용한다.

| 컨트롤 | 기본 | 의미 |
| --- | --- | --- |
| `roll augment` | 켬 | 카메라를 광축 둘레로 돌린 자세도 후보에 넣는다. 검사 결과는 같으므로 **닿는 자세가 늘어난다** |
| `roll-step-deg` | 30.0 | 몇 도 간격으로 돌려 볼지 |
| `tilt augment` | 켬 | 카메라를 살짝 기운 자세도 후보에 넣는다 |
| `tilt-angles-deg` | 5 10 | 기울일 각도들 (공백으로 구분) |
| `tilt-azimuths` | 8 | 기울이는 방향을 몇 갈래로 나눌지 |
| `dedup` | 켬 | 거의 같은 해를 하나로 합친다 |
| `dedup-rad` | 0.08 | 이 값 안쪽이면 같은 해로 본다 (rad) |
| `num-seeds` | 32 | 한 지점당 IK를 몇 번 다른 시작값으로 풀지. **키우면 더 많이 찾지만 느려진다** |
| `ik-batch-size` | 128 | GPU에서 한 번에 푸는 개수. 결과에는 영향이 없고 속도·메모리만 바뀐다 |

| 버튼 | 동작 |
| --- | --- |
| `Check and Save IK` | 위 설정으로 IK를 풀고 결과를 저장한다. 몇 개가 닿고 몇 개가 안 닿는지 `Log`에 뜬다 |
| `Cancel IK Check` | 도는 중에 멈춘다 |

> 💡 **옵션을 바꾸면 다시 푼다**
>
> `Generate Scan Motion`은 저장된 IK를 재사용하는데, **이 패널의 옵션이 그대로일 때만** 그렇다. 하나라도 바꾸면 경로 생성이 IK부터 다시 푼다 — 그만큼 오래 걸린다.

> ⚠️ **주의 · 닿지 않는 지점이 있다면**
>
> 물체 위치를 조금 옮기거나 돌려 보면 크게 달라진다. 도달 가능한 지점 수는 배치에 매우 민감하다. `Load Object`로 다시 놓고 `Check and Save IK`를 다시 눌러 비교한다.

### 5 · Motion Speed

앞으로 만들 경로가 **얼마나 빠르게 실행될지**를 정한다. 이 값들은 경로를 만드는 순간 파일에 함께 구워지므로, **경로를 만들기 전에** 정해야 한다. 이미 만든 경로는 예전 속도를 그대로 갖고 있다.

| 컨트롤 | 기본 | 의미 |
| --- | --- | --- |
| `scan EE speed [mm/s]` | 10 | 검사 구간에서 카메라가 움직이는 속도 |
| `scan EE angular [deg/s]` | 20 | 검사 구간에서 카메라 방향이 돌아가는 속도 |
| `max joint vel [rad/s]` | 0.3 | 관절 속도 상한. **이동 구간의 속도는 이 값만으로 정해진다** |
| `min segment dt [s]` | 0.05 | 구간 하나에 최소한 들이는 시간. 짧은 구간이 순간적으로 빨라지는 것을 막는다 |
| `corner threshold [deg]` | 30 | 이보다 급하게 꺾이는 곳부터 감속한다 |
| `corner max slowdown [x]` | 2.5 | 가장 급한 꺾임에서 몇 배 느려질지. `1`이면 감속하지 않는다 |

> 💡 **셋 중 가장 느린 것이 이긴다**
>
> 검사 구간은 **카메라 속도 · 카메라 회전 속도 · 관절 속도**를 모두 만족하는 시간으로 계산된다. 그래서 카메라 속도를 3배로 올려도 관절 속도에 걸리면 3배가 되지 않는다. 이동 구간은 검사가 아니므로 관절 속도만 본다.
>
> `0`을 넣으면 그 제한 하나를 끄는 뜻이다. 음수는 받지 않는다.

### 6 · Generate Trajectory

실제 실행할 경로를 만든다. 두 종류를 만들 수 있다.

#### Scan options (GLNS) — 검사 경로

모든 viewpoint를 한 번씩 도는 경로다. 방문 순서와 각 지점에서의 팔 자세를 함께 최적화하고, 지점 사이를 충돌 없이 잇는다.

| 컨트롤 | 기본 | 의미 |
| --- | --- | --- |
| `--delaunay-expand-hops` | 2 | 어느 정도 떨어진 지점끼리 이어도 되는지. 키우면 더 좋은 순서를 찾을 수 있지만 느려진다 |
| `--max-candidates-per-viewpoint` | 32 | 한 지점당 고려할 팔 자세 후보 수 |

| 버튼 | 동작 |
| --- | --- |
| `Generate Scan Motion` | 검사 경로를 만든다. 끝나면 결과 파일 경로가 아래 `CSV path` 칸에 자동으로 들어가고 미리보기에 올라간다 |
| `Cancel` | 도는 중에 멈춘다 |

> 💡 **몇 분 걸린다**
>
> viewpoint 수와 옵션에 따라 다르지만 보통 수십 초에서 몇 분이다. 도는 동안 다른 버튼은 잠기고 `Cancel`만 살아 있다. 진행 상황은 `Log`에 계속 흘러나온다.

#### Tilt options — 한 지점 정밀 관찰

검사 경로와 달리, **한 지점을 중심으로 카메라를 상하좌우로 기울여 가며 훑는** 경로다. 특정 부위를 여러 각도에서 자세히 볼 때 쓴다.

| 컨트롤 | 기본 | 의미 |
| --- | --- | --- |
| `base trajectory` | — | 중심으로 삼을 지점을 **어느 경로에서 셀지**. `Browse...`로 고른다. 아래 Preview·Execute의 `CSV path`와는 별개 칸이다 |
| `center row idx` | 0 | 그 경로의 몇 번째 지점을 중심으로 쓸지 |
| `Highlight` | — | 고른 지점을 화면에 노란 점으로 표시한다 |
| `pitch down/up deg` + `n` | −20 / 20 / 40 | 위아래로 기울일 각도 범위와 샘플 수 |
| `roll left/right deg` + `n` | −20 / 20 / 40 | 좌우로 기울일 각도 범위와 샘플 수 |
| `num-seeds` / `ik-batch-size` | 32 / 128 | Solve IK의 같은 이름 옵션과 같은 뜻 |
| `clamp unreachable angles` | 켬 | 요청한 각도까지 팔이 안 닿으면 **닿는 데까지만** 하고 넘어간다. 끄면 실패로 처리한다 |

| 버튼 | 동작 |
| --- | --- |
| `Generate Tilt Motion` | tilt 경로를 만든다 |
| `Show Tilt Fan` / `Clear Tilt Fan` | 만들기 전에 **어떤 부채꼴을 그릴지 화면에 미리 그려 본다.** 각도를 바꾸면 그림이 바로 따라 바뀐다 |

> 💡 **중심 지점의 종류가 표시된다**
>
> `center row idx` 아래 줄에 `row 60 / 271 planned` 처럼 뜬다. 뒤의 단어가 그 지점의 성격이다.

| 표시 | 뜻 |
| --- | --- |
| `viewpoint` | 검사 지점 그 자체 — 실제로 사진을 찍는 자리 |
| `interpolated` | 두 검사 지점 사이를 부드럽게 잇느라 생긴 중간 점 |
| `planned` | 장애물을 피해 옮겨 가느라 경로 계획기가 만든 점 |

> 📷 **PLATE 10 · 사진**
>
> `Show Tilt Fan`을 눌러 물체 위에 부채꼴 궤적이 그려진 뷰포트. 상하 방향과 좌우 방향이 색으로 구분되어 보이는 각도.

### 7 · Preview in Simulation

만든 경로를 **반투명 고스트 로봇**으로 재생해 본다. 진짜 로봇은 전혀 움직이지 않으므로 안전하고, Simulation·Real 어느 모드에서도 쓸 수 있다.

| 컨트롤 | 동작 |
| --- | --- |
| `CSV path` | 재생·실행할 경로 파일. 경로를 만들면 자동으로 채워진다. `Browse...`로 다른 파일을 고를 수도 있다 |
| `Load & Preview` | 그 파일을 고스트에 올린다 |
| `Play` / `Pause` / `Stop` | 재생 · 일시정지 · 처음으로 |
| `t` 슬라이더 | 원하는 시점으로 직접 끌어 본다 |
| `Show Collision Spheres` / `Clear` | 충돌 검사에 쓰이는 로봇의 구를 그린다. **계획기가 로봇을 어떤 덩어리로 보는지** 확인하는 용도 |
| `FOV W` / `FOV H` / `WD` | 고스트에 붙은 카메라의 화각·거리 값 |
| `Reset` | 불러온 viewpoint 파일의 값으로 되돌린다 |
| `Show FOV` | 카메라가 담는 사각형을 화면에 그린다 |
| `Show Camera Range` | 카메라에서 뻗는 거리 선을 그린다. 물체까지 실제 거리를 눈으로 확인할 수 있다 |

패널 맨 아래 상태 줄이 **지금 무엇이 재생 중인지**를 항상 적는다.

**화면 · 상태 줄 읽는 법**

```
t=2.14s / 6.31s  (wp 140/411)  playing: home_move_approach.csv
                                        └ 지금 고스트가 트는 파일

CSV path: .../trajectory/74/trajectory.csv
          └ Execute Scan 이 실행할 파일
```

---

> 📷 **PLATE 11 · 영상**
>
> `Load & Preview` → `Play`로 고스트가 검사 경로를 훑는 20 – 30초 클립. `Show FOV`를 켜 두어 카메라 화각이 물체 표면을 덮으며 지나가는 것이 보이면 좋다.

### 8 · Execute Trajectory

여기서부터는 로봇이 진짜로 움직인다. 구조는 단순하다 — **먼저 계획해서 눈으로 보고, 그 다음에 움직인다.**

| 버튼 | 동작 |
| --- | --- |
| `Plan to Start` | 지금 자세에서 검사 경로 시작점까지 **충돌 없는 이동을 계획**하고, 고스트로 한 번 재생한다. 로봇은 움직이지 않는다 |
| `Move to Start` | 그 계획을 실행한다. **계획이 없으면 눌리지 않는다** |
| `Plan to HOME` | 지금 자세에서 대기 자세까지의 이동을 계획한다 |
| `Move to HOME` | 그 계획을 실행한다 |
| `Log Joints` | 지금 로봇의 관절 각도를 `Log`에 찍는다. 어디로 갈지 정하기 전에 **어디에 있는지** 읽는 자리 |
| `Execute Scan` | `CSV path`의 경로를 실행한다 |
| `Cancel` | 계획 중이든 실행 중이든 멈춘다 |

> 💡 **Execute Scan은 아무 때나 눌리지 않는다**
>
> 지금 로봇 자세가 검사 경로의 첫 지점과 **같을 때만** 활성화된다. 다르면 버튼이 흐려지고 옆에 이유가 적힌다 — `robot is 42.3 deg from the scan start`. `Plan to Start` → `Move to Start`를 먼저 하라는 뜻이다.
>
> 이 잠금이 없으면, 실행기가 현재 자세와 첫 지점 사이를 **미리 본 적 없는 직선**으로 메우며 움직인다. 자세에 따라 팔이 한 바퀴 도는 일도 생긴다.

> 💡 **계획이 낡으면 거부한다**
>
> `Move`는 누르는 순간 계획 당시의 조건을 다시 확인한다 — 로봇이 아직 그 출발점에 있는지, 목표와 물체 위치가 그대로인지. 하나라도 어긋나면 **움직이지 않고** 이유를 `Log`에 적은 뒤 계획을 버린다. 다시 계획하면 된다.

> ⚠️ **주의 · 경로를 못 찾을 때**
>
> `plan exit code = 2`가 뜨면 충돌 없는 길을 못 찾은 것이다. 물체 위치를 조금 옮기거나 로봇 자세를 바꿔 다시 시도한다. 그래도 안 되면 `Pipeline Mode`를 MoveIt으로 바꿔 RViz에서 손으로 옮기는 방법이 있다.

#### 표준 작업 순서

1. Simulation 또는 mock hardware로 먼저 전 과정을 확인한다.
2. `Load & Preview` → `Play`로 경로를 끝까지 본다.
3. `Plan to Start` — 고스트가 접근 경로를 재생한다. 주변과 부딪히지 않는지 본다.
4. `Move to Start` — 로봇이 시작점으로 간다.
5. `Execute Scan` — 검사 구간을 실행한다.
6. 끝나면 `Plan to HOME` → 확인 → `Move to HOME`.

> 📷 **PLATE 12 · 영상**
>
> 표준 순서 전체를 담은 1분 클립. Plan to Start(고스트) → Move to Start(실물) → Execute Scan → Move to HOME. 화면에 패널과 뷰포트가 같이 보이게.

> 🤖 **실로봇 · 실행 전 확인**
>
> 로봇 주변과 케이블이 지나갈 범위가 비어 있는지 확인한다. 비상 정지 수단을 손이 닿는 곳에 둔다. `Motion Speed`를 낮춰 처음 한 번은 천천히 돌린다. `Cancel`이 눌리는 상태인지 미리 확인한다.

### 보조 패널

| 패널 / 컨트롤 | 용도 |
| --- | --- |
| `Scene (obstacles)` | 지금 쓰는 워크셀 정의 파일 이름을 보여준다. 평소엔 접어 둔다 |
| `Log Selected Prim as YAML` | 화면에서 고른 물체의 위치·크기를 워크셀 정의에 붙여 넣을 형식으로 `Log`에 찍는다. **파일을 고치지는 않는다** — 실제 셀 치수를 재서 반영할 때만 쓴다 |
| `Log` | 모든 작업의 출력이 여기로 흘러나온다. 무언가 실패하면 **가장 먼저 볼 곳** |

> ⚠️ **주의 · 화면에서 옮긴 것은 계획에만 반영된다**
>
> 뷰포트에서 **장애물**을 기즈모로 옮겨도 워크셀 정의 파일은 바뀌지 않는다. 다음에 앱을 켜면 원래 자리로 돌아온다. 영구히 반영하려면 `Log Selected Prim as YAML`로 숫자를 뽑아 파일에 적어야 한다. (**검사 대상 물체**는 예외 — 옮긴 위치가 그대로 경로 생성에 쓰인다.)

---

## 부록

### 파일이 저장되는 곳

| 단계 | 경로 |
| --- | --- |
| 물체 메시 | `data/{물체}/mesh/source.obj` |
| Viewpoint | `data/{물체}/viewpoint/{N}/viewpoints.h5` |
| IK 해 | `data/{물체}/ik/{N}/ik_*.h5` |
| 방문 순서 | `data/{물체}/trajectory/{N}/solution.h5` |
| 검사 경로 | `data/{물체}/trajectory/{N}/trajectory.csv` |
| Tilt 경로 | `data/{물체}/trajectory/{N}/trajectory_tilt_vp{번호}.csv` |
| 이동 경로 | `data/{물체}/trajectory/{N}/home_move_*.csv` |
| 워크셀 정의 | `workcell/scenes/{이름}.yaml` |

### 경로 파일 안에 무엇이 들어 있나

`.csv`는 로봇이 그대로 따라가는 표다. 한 줄이 한 시점이고, 열은 이렇게 구성된다.

| 열 | 내용 |
| --- | --- |
| `time` | 시작부터 이 지점까지 걸리는 시간(초). **이 열이 곧 실행 속도다** |
| `ur20-*_joint` | 여섯 관절의 각도(rad) |
| `target-POS_*` / `target-ROT_*` | 그때 카메라의 위치와 방향 |
| `waypoint_kind` | 그 지점의 성격 — `viewpoint` / `interpolated` / `planned` |

같은 이름의 `.npz`가 옆에 함께 저장된다. 어떤 설정으로 만든 경로인지(물체 위치, 작업 거리, 속도 값)가 거기 남는다.

> 💡 **산출물은 버전 관리에 들어가지 않는다**
>
> `data/**/viewpoint/` · `data/**/trajectory/` · `data/**/ik/` 는 `.gitignore`에 있다. 저장소를 새로 받으면 이 폴더들이 비어 있는 것이 정상이고, 각자 만들어 쓴다. 버전 관리되는 것은 **메시와 워크셀 정의**뿐이다.

### 자주 겪는 문제

| 증상 | 원인과 조치 |
| --- | --- |
| Isaac 창이 안 뜬다 | 호스트에서 `xhost +local:root`를 다시 실행한다. 재부팅하면 풀린다 |
| `CUDA driver version is insufficient` | NVIDIA 드라이버가 580 미만이다. 드라이버를 올린다 |
| `Switch controller timed out` | Isaac에서 **▶ Play**를 안 눌렀다. 누르고 ROS 스택을 다시 실행한다 |
| ROS 노드가 갑자기 죽는다 | sim 스택과 real 스택을 동시에 띄웠다. 하나만 남긴다 |
| `Execute Scan`이 흐리다 | 로봇이 시작점에 없다. `Plan to Start` → `Move to Start` |
| `Move`가 안 눌린다 | 계획이 없거나 낡았다. `Plan`을 먼저 누른다 |
| `plan exit code = 2` | 충돌 없는 길이 없다. 물체나 로봇 자세를 바꿔 다시 시도한다 |
| viewpoint가 많이 안 닿는다 | 물체 배치를 바꿔 본다. 도달성은 배치에 매우 민감하다 |
| 속도를 바꿨는데 그대로다 | 속도는 경로를 **만들 때** 구워진다. 경로를 다시 생성한다 |

### 용어

| 용어 | 뜻 |
| --- | --- |
| Viewpoint | 검사할 때 카메라가 서야 할 위치와 바라볼 방향 |
| Working distance | 표면에서 카메라까지의 거리. 초점이 맞는 거리 |
| FOV | 그 거리에서 카메라가 담는 실제 크기 (가로 × 세로, mm) |
| IK | 원하는 카메라 자세를 만들기 위한 여섯 관절 각도를 푸는 것 |
| Waypoint | 경로를 이루는 한 지점. 경로는 waypoint의 나열이다 |
| Transit / 이동 구간 | 검사가 아니라 다음 검사 지점으로 옮겨 가는 구간 |
| Ghost / 고스트 | 미리보기용 반투명 로봇. 진짜 로봇을 건드리지 않는다 |
| HOME | 정해진 대기 자세. 작업 시작과 끝의 기준점 |

# Architecture

## 파이프라인 흐름

```
Stage 1: 뷰포인트 생성 + 경로 최적화
  generate_viewpoints.py [--cluster]
  Input:  data/{object}/mesh/source.obj (+ optional material RGB 필터)
  Output: data/{object}/viewpoint/{num}/viewpoints.h5
  Process: PCA 주축 → 그리드 샘플링 → 표면 법선 계산 → zigzag path order
           [--cluster] 클러스터링 → 클러스터 내 지그재그 + 클러스터 간 greedy NN

Stage 2: 궤적 생성 (scripts/prev/2_generate_trajectory.py)
  Input:  viewpoints.h5, mesh/source.obj
  Output: data/{object}/trajectory/{num}/trajectory.csv
  Process:
    2a. 카메라 포즈 계산 (위치 + 법선 → 4x4 변환행렬)
    2b. IK 풀이 (cuRobo, goalset: z축 회전 변형)
    2c. 충돌 검사 (SDF 기반, 메시 + 큐보이드)
    2d. GTSP 최적 순서 (cuOpt, joint-space 거리 행렬)
    2e. 선형 보간 + 충돌 재검사
    2f. 충돌 구간 재계획 (cuRobo trajopt, max 60회 시도)
    2g. CSV 저장 (joint angles in radians)

Stage 3: 시뮬레이션 (scripts/prev/3_simulation.py)
  Input:  trajectory.csv
  Runtime: Isaac Sim (omni_python)

Stage 4: 로봇 실행
  4_publish_trajectory.py (prev/) 또는 plan_motion.py --publish
  ROS2 FollowJointTrajectory action → ur_robot_driver
```

## 데이터 형식

### viewpoints.h5
```
viewpoints/
  positions  (N, 3)  float64  표면 좌표 (object local frame, meters)
  normals    (N, 3)  float64  표면 법선
  path_order (N,)    int32    zigzag 순서 인덱스
metadata/
  camera_spec/
    working_distance_mm  float
```

### trajectory.csv
```
shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
-1.5708, -2.0944, -1.0472, ...
```
각 행이 하나의 waypoint (radians).

## 카메라-로봇 변환 체인

```
표면 위치 + 법선
  → 카메라 위치 (표면 + 법선 × working_distance)
  → 카메라 포즈 (approach = -법선, z축 회전 자유)
  → object frame → world frame 변환 (TARGET_OBJECT rotation/position)
  → cuRobo IK → 6-DOF joint angles
```

## IK 전략 (plan_motion.py)

1. **Goalset IK**: 뷰포인트당 N개의 z축 회전 변형 (마지막 조인트 자유도 활용)
2. **Multi-seed**: `return_seeds=32`로 뷰포인트당 32개 IK 후보 생성
3. **Smooth selection** (`select_smooth_trajectory()`):
   - path_order 순서로 greedy nearest-neighbor 선택
   - 시작점: `config.ROBOT_START_STATE`
   - 거리 기준: L-inf (최대 joint 변화량)
   - 연속 뷰포인트 간 joint jump 최소화

## 궤적 전송 (--publish 모드)

`FollowJointTrajectory` action으로 `scaled_joint_trajectory_controller`에 전송.

1. **현재 위치 읽기**: `/joint_states` 구독 (이름 매칭으로 순서 무관)
2. **보간**: 연속 waypoint 간 max_step 0.1 rad 이내로 선형 보간
3. **시간 할당**: 각 sub-step의 시간 = max_joint_diff / MAX_JOINT_VEL (1.0 rad/s)
4. **t=0 포인트**: 현재 로봇 위치를 첫 번째 포인트로 포함 (tolerance violation 방지)

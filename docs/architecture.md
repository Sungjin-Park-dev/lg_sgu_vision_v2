# Architecture

## 파이프라인

```
Stage 1: 뷰포인트 생성 + 클러스터링 + 클러스터 순서 최적화
  scripts/generate_viewpoints.py
  Input:  data/{object}/mesh/target.{ply,obj} (+ optional material RGB 필터)
  Output: data/{object}/viewpoint/{num}/viewpoints_{method}.h5 + .html
  Process: PCA 주축 → 그리드 샘플링 → 표면 투영/법선 계산
           → 클러스터링(dbscan/coacd/coacd+dbscan)
           → 클러스터 내부: PCA zigzag (grid row_index 기반)
           → 클러스터 간: GTSP (Noon-Bean + OR-Tools ATSP)

Stage 2: 궤적 생성
  scripts/plan_motion.py
  Input:  viewpoints.h5 (클러스터 데이터 필수)
  Output: data/{object}/trajectory/{num}/trajectory.csv + trajectory.html
  Process:
    - 클러스터 내부: dense Cartesian 보간 + seed-propagation IK (cuRobo)
    - 클러스터 간: MotionGen 충돌 회피 이동
    - 충돌 검사 (SDF: 메시 + 큐보이드)

Stage 3: 로봇 실행 (ROS2)
  scripts/publish_trajectory.py
  Input:  trajectory.csv
  Output: FollowJointTrajectory action → ur_robot_driver
```

## 데이터 형식

### viewpoints.h5

```
viewpoints/
  positions      (N, 3)  float32  표면 좌표 (object local frame, meters)
  normals        (N, 3)  float32  표면 법선
  path_order     (N,)    int32    최종 방문 순서 인덱스
  row_index      (N,)    int32    grid 행 인덱스 (zigzag 계산용)
  cluster_id     (N,)    int32    각 뷰포인트의 클러스터 할당
  cluster_order  (K,)    int32    클러스터 방문 순서 (GTSP 결과)
  pca_center/axis1/axis2

metadata/
  camera_spec/working_distance_mm
  clustering_method         str    "dbscan" | "coacd" | "coacd+dbscan"
  dbscan_eps_mm / coacd_threshold 등 파라미터
  num_clusters              int
```

### trajectory.csv

```
shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
-1.5708, -2.0944, -1.0472, ...
```
각 행이 하나의 waypoint (radians). `JOINT_NAMES` 순서 고정.

## 카메라-로봇 변환 체인

```
표면 위치 + 법선
  → 카메라 위치 = 표면 + 법선 × working_distance
  → 카메라 포즈 (approach = -법선, z축 회전 자유)
  → object frame → world frame (TARGET_OBJECT rotation/position)
  → cuRobo IK → 6-DOF joint angles
```

## 궤적 전송 (publish_trajectory.py)

별도 스크립트로 분리되어 있으며 `/scaled_joint_trajectory_controller/follow_joint_trajectory` action으로 전송.

1. **현재 위치 읽기**: `/joint_states` 구독 (이름 매칭으로 조인트 순서 무관)
2. **보간**: 연속 waypoint 간 `MAX_STEP_RAD=0.1` 이내로 선형 보간
3. **시간 할당**: 각 sub-step 시간 = `max_joint_diff / MAX_JOINT_VEL` (기본 2.0 rad/s)
4. **t=0 포인트**: 현재 로봇 위치를 첫 포인트로 포함 (tolerance violation 방지)

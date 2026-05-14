# select_ik_dp.py

viewpoints.h5의 각 뷰포인트에 대해 IK를 풀고, DBSCAN + DP로 전역 최적 joint trajectory를 생성한다.

## 파이프라인

```
viewpoints.h5 → Multi-seed IK → DBSCAN 대표 추출 → DP 최적 경로
→ wrist_3 고정 → MotionGen transit (reconfig 지점)
→ Uniform resample + 충돌 검사 → Continuous-scan time planning → trajectory_dp.csv + trajectory_dp.html
```

## Phase 1: Multi-seed IK

viewpoint당 100개 seed로 cuRobo IK를 풀어 다수의 후보 해를 생성.

- 출력: `(N, num_seeds, 6)` joint solutions + success mask
- IK 해는 즉시 `normalize_joints()` 적용 — `[-π, π]`로 정규화
  - -270°와 +90°처럼 물리적으로 같은 자세가 다른 값으로 되는 2π oscillation 방지
- 전체 성공률 ~37% (sample 124 viewpoints 기준)

## Phase 2: DBSCAN (viewpoint당)

각 viewpoint의 성공한 IK 해를 joint-space에서 DBSCAN으로 클러스터링하고, 클러스터마다 medoid(중심에 가장 가까운 해) 하나를 대표로 추출.

- `eps=0.3` rad, `min_samples=1`
- noise 포인트(-1)는 각각 singleton으로 취급
- 평균 ~3.9개 대표/viewpoint
- Phase 3 DP의 탐색 공간을 줄이는 역할

## Phase 3: DP 최적 경로

N개 viewpoint × 각 K_i개 대표 해에서 전역 최적 경로를 선택.

비용 함수: `cost = is_reconfig × 1000 + L2_distance`
- **1순위**: reconfig 최소화 (angular L-inf > threshold이면 reconfig, 페널티 1000)
- **2순위**: joint-space L2 거리 최소화
- reconfig threshold: 기본 29°

빈 viewpoint(IK 성공 해 없음)는 인접 viewpoint의 해를 carry-forward.

DP 후 클러스터 간/내 reconfig 분석:
- inter-cluster reconfig: 클러스터 전환 시 발생 — 예상되는 동작
- intra-cluster reconfig: 같은 클러스터 내에서 발생 — 이상적으로는 0

## wrist_3 고정 (Phase 3 직후)

DP 결과 `selected[:, -1]`을 `ROBOT_START_STATE[-1]`로 잠금. 카메라 z축 회전 자유도는 검사 품질에 무관하므로 고정해도 EE 정확도 무관, 다만 *resample 이전에* 잠궈야 인접 row의 L2 spacing이 균일해진다 (모든 waypoint의 wrist_3가 동일 상수면 6-DoF L2 = 5-DoF L2).

## Phase 4: MotionGen transit

Reconfig 지점(큰 joint jump)마다 cuRobo MotionGen으로 충돌 회피 경로 생성.

- joint-to-joint planning (`plan_single_js`)
- 성공 시 중간 waypoints 삽입, 실패 시 원래대로 직접 점프
- `max_attempts=10`, `timeout=5.0s`
- Plan 후 transit_segments[:, -1]도 wrist_3 강제 (MotionGen이 흔들었을 경우 안전망)

## Phase 5: Uniform resample + 충돌 검사

1. **보간**: non-reconfig 구간은 joint-space 선형 보간, transit 구간은 MotionGen 경로 그대로
2. **Uniform resample**: EE arc-length 기준 등간격(기본 10mm).
3. **Batch collision check**: 자기 충돌 + 환경 충돌. 충돌 waypoint 제거 후 cumulative L2 arc-length로 재resample하여 균일성 복원.

## Continuous-scan time planning

최종 CSV에는 실제 실행용 `time` 컬럼이 저장된다.

- 시간 배분은 EE 선속도, EE 각속도, joint 속도 제한을 모두 만족하도록 각 segment의 최대 필요 시간을 사용한다.
- EE 또는 joint-space 방향 전환이 큰 waypoint 주변 segment는 추가로 감속한다.
- `publish_trajectory.py`는 CSV의 `time`을 그대로 ROS trajectory time으로 사용한다.

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--object` | (필수) | Object name |
| `--num-viewpoints` | (필수) | 뷰포인트 수 |
| `--viewpoints` | None | h5 파일 직접 지정 |
| `--spacing` | 0.01 | Phase 5 EE resample 간격 (m) |
| `--output-suffix` | `dp` | 출력 파일 접미사 |

고급 튜닝값(IK seed 수, DBSCAN eps, reconfig threshold, EE 선속도/각속도,
joint 속도 제한, corner slowdown)은 `scripts/pipeline/select_ik_dp.py` 상단의
`Pipeline defaults` 상수에서 관리한다.

## 출력

| 파일 | 내용 |
|------|------|
| `trajectory_{suffix}.csv` | `time` + joint angles + EE pose (robot name prefix 포함) |
| `trajectory_{suffix}.html` | static 3D 시각화: 메시 + EE 경로 + reconfig 표시 |
| `trajectory_{suffix}_anim.html` | animated: 슬라이더/Play로 step별 경로 성장 |

속도 계획 옵션은 파일명에 포함된다. 예:
`trajectory_dp_ee_s0010_eev50mms_av20dps_jv0p30_corner30d_x2p5.csv`

## 실행 예시

```bash
uv run scripts/pipeline/select_ik_dp.py \
  --object sample --num-viewpoints 124 \
  --viewpoints data/sample/viewpoint/124/viewpoints_coacd+dbscan.h5
```

```
Phase 1: 4664/12400 IK solutions (37.6% success)
Phase 2: avg 3.9 representatives/viewpoint (eps=0.30 rad)
Phase 3: 15 reconfigs, max_jump=147.1°, mean_jump=13.3°
  Inter-cluster: 7, Intra-cluster: 8 (cluster 13)
Phase 4: MotionGen transit for 15 reconfig points
Phase 5: uniform resample → collision check → final trajectory
```

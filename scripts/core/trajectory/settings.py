"""Trajectory planning defaults shared by DP and GLNS pipelines."""

import numpy as np

from common import config

ROBOT_CONFIG = config.DEFAULT_ROBOT_CONFIG
NUM_IK_SEEDS = 100
IK_BATCH_SIZE = 4
IK_RANDOM_SEED = 123
RECONFIG_THRESHOLD_DEG = 29.0

# IK 후보: 성공 IK 해 전체를 fine tolerance로 near-duplicate만 제거하고 모든 분기를 후보로 남겨
# 순서 최적화(GLNS)가 이웃과 연속인 분기를 직접 고르게 한다(클러스터 medoid는 연속 해를 버릴 수
# 있어 미사용).
CANDIDATE_DEDUP_RAD = 0.08   # ~4.6°, 분기는 보존하며 거의 동일한 seed 만 제거

# Reconfig transit(충돌회피 joint-to-joint) 계획.
# plan_cspace는 timeout이 없고 max_attempts 회 재시도 후 실패하는 단순 루프라(성공 시 즉시 break),
# '실패 판정에 걸리는 시간'이 max_attempts에 거의 선형 비례한다(이 하드웨어에서 ~0.33s/attempt).
# 성공은 attempt 0~1(~0.37s)에 끝나므로 max_attempts를 줄여도 성공은 거의 영향 없고 실패 대기만 짧아진다.
# transit 은 trajopt-only(직선 시드)로 계획한다. graph(PRM) seeding 은 끈다 — 빡센 reconfig
# (측면 flip 등 직선 시드가 물체 관통)는 via-roll/via-tilt 가 rolled 중간자세 경유로 흡수하므로
# graph 가 redundant 하고 PRM 이 느림. graph off + via-roll + batch(아래)로 coverage 동일·대폭 단축
# (curved 99/100 21s, sample 69/74 28s, collisions=0; graph 켜던 원본 대비 ~5x). graph 없으면 성공은
# attempt 0 에 끝나 실패만 max_attempts 만큼 헛도므로 max_attempts 도 낮게 둔다. (필요 시 graph 재활성: ATTEMPT=1)
TRANSIT_MAX_ATTEMPTS = 1
TRANSIT_ENABLE_GRAPH_ATTEMPT = 99  # max_attempts 밖으로 설정해 graph(PRM) 비활성화

# transit 은 BatchMotionPlanner 로 후보 leg 를 GPU 배치 병렬 계획한다(경계내 순차 탐색 제거).
# 한 plan_cspace 호출이 batch_size 문제를 동시에 푼다 — direct 는 전 경계를 한 배치로, via-roll/tilt 는
# 경계내 후보(leg + bridge)를 joint-closest 순 chunk 로 풀되 가교쌍이 나오면 즉시 멈춘다(short-circuit).
# batch_size 는 한 경계의 후보(leg 2*MAX_REPS + bridge (MAX_REPS+1)²-1 ≈ 8+24)가 한 chunk 에 들면 빠르다.
# plan_cspace 1 chunk ≈ 2.2s(compute-bound). 작으면 via-roll 경계가 2 chunk 로 쪼개져 생성 약간↑.
# ★ 메모리(2026-06-29 probe_planner_vram.py 실측): BatchMotionPlanner warmup 의 trajopt 버퍼가 batch_size 에
#   ~선형(≈375 MB/slot, build 자체는 11 MB) → peak reserved bs8≈3.0 / bs16≈6.0 / bs24≈9.0 / bs32≈12 GB.
#   Phase-1 IK 는 ~200 MB 로 무관(과거 "IK 11 GB" 기록은 오귀속 — 진짜 hog 는 이 planner warmup).
#   16 GB GPU 에 Isaac Sim(~3.8 GB) 공존 시 bs32(12 GB)는 OOM, bs24(9 GB)는 fit → **24** 채택.
# ★ batch 축소 안전성: bs 를 낮추면 plan_cspace batch-composition 이 바뀌어 transit 궤적이 달라질 수 있고
#   (결정적이나 baseline 과 다름), 과거엔 그게 grazing transit 을 만들어 최종 collision 검증을 깨뜨렸다.
#   이제 via-roll/direct 가 채택 전 `_transit_safe`(densify-검증, 최종과 동일 기준)로 grazing 후보를 거부
#   하므로 bs 가 뭐든 **항상 유효한 transit 만** 선택 → bs 가 안전한 메모리/속도 노브가 됐다.
TRANSIT_BATCH_SIZE = 24

# transit이 끝내 실패한 reconfig를 직선 보간으로 메우면 카메라/팔이 물체를 관통한다.
# 대신 아웃라이어 viewpoint를 최대 이 개수까지 건너뛰어(skip), 시작 자세와 다시
# reconfig_threshold 안에서 만나는 viewpoint에 재연결한다.
TRANSIT_FAIL_SKIP_MAX = 5

# via-roll: 직접 transit 실패 시, 경계의 양 끝 scan 자세를 광축(camera 광축 ≈ wrist_3 축) 둘레로
# roll 한 '중간자세'를 경유해 direct 가교한다. wrist_3 lock 이 버린 redundant DOF(광축 roll =
# 검사 무손실)를 경계 transit 에서만 복원하는 것 — scan config(selected[i]) 는 보존하므로 검사
# 품질·스캔 일관성에 영향 없다(중간자세는 스캔되지 않음). transit-fail 의 주 해결 레버.
ROLL_VARIANT_DEG = (45, 90, 135, 180, 225, 270, 315)   # 광축 둘레 roll 변형 각도
VIA_ROLL_IK_SEEDS = 40            # 변형 타깃당 IK seed 수 (collision-free 분기 확보)
VIA_ROLL_MAX_REPS = 4             # endpoint 당 변형 후보 상한. bridge 쌍 ≤(MAX_REPS+1)²; TRANSIT_BATCH_SIZE
                                  # 와 짝지어 한 경계가 1 chunk 에 들게 함(4→32). 키우면 coverage↑·속도↓

# via-tilt: roll 로도 못 푼 경계를 표면점 중심 orbit tilt(광축 ±φ)로 escalation. 중간자세는
# 스캔되지 않으므로 시야 비스듬함은 무비용. (wd_m 필요)
TILT_VARIANT_PHI = (15, 30)       # 광축 기울임 각도(도)
TILT_VARIANT_AZ = (0, 90, 180, 270)

# 큰 base reconfig(어깨/팔꿈치 분기 flip)은 via-roll/via-tilt 가 광축만 건드려 절대 못 가교한다
# → 곧장 via-home 으로(채택 route·실행 모션 동일, 헛된 attempt 만 생략 = 생성시간만 단축).
BIG_BASE_RECONFIG_RAD = np.deg2rad(150.0)

# 충돌검사에서 제외할 로봇 링크. base_link_inertia(로봇 베이스)는 base_link 에 고정이라
# 자세와 무관하게 항상 robot_mount(받침대) 윗면을 ~2cm 파고든다 → 모든 IK/충돌검사가
# 상시 충돌로 실패. 받침대 박스가 팔은 그대로 막아주고, base 는 자기 받침대만 닿으므로
# 충돌검사 자체가 무의미해 제외해도 실질 보호 손실이 없다.
COLLISION_EXCLUDE_LINKS = ("base_link_inertia",)

RESAMPLE_MODE = "ee"
DEFAULT_SPACING_M = 0.01

# ── 실행 속도 노브 ────────────────────────────────────────────────────────────
# 아래 6개(+ 코너 2개)가 CSV 의 time 열을 정하고, **그 열이 곧 실행 속도다** — Isaac 고스트
# 재생도 실로봇 publish 도 이 열을 그대로 따른다. 값은 생성 시점에 궤적에 구워지므로 바꾸면
# 다시 생성해야 한다(CSV 를 나중에 다시 시간매김하는 경로는 없다).
#
# collision_gate_and_save 는 이 값들을 **모듈 속성으로 호출 시점에** 읽는다. 그래서
# apply_timing_cli() 로 여기를 덮어쓰면 그대로 걸린다 — `from .settings import EE_SPEED_MM_S`
# 처럼 값을 복사해 가면 override 가 조용히 무시되니 하지 말 것.
EE_SPEED_MM_S = 10.0
EE_ANGULAR_SPEED_DEG_S = 20.0
MAX_JOINT_VEL_RAD_S = 0.3
MIN_SEGMENT_DT_S = 0.05

# reconfig transit(재배치)은 검사 스캔이 아니라 단순 repositioning이다. 스캔과 똑같이 EE
# arc-length로 resample하고 EE 선속도(50mm/s)로 시간 매기면, base를 크게 돌릴 때 팔 끝이
# 자유공간에 그리는 긴 호(수 m)를 기어가느라 사이클의 대부분을 먹는다. 그래서 transit 구간은
# (1) joint-space L∞로 sparse하게 resample하고, (2) joint 속도 한계로만 시간을 매긴다
# (EE 선속도/각속도/corner slowdown 무시).
TRANSIT_RESAMPLE_SPACING_RAD = 0.05   # ~2.9°, transit resample 간격(가장 빨리 도는 joint 기준)

CORNER_ANGLE_THRESHOLD_DEG = 30.0
CORNER_MAX_SLOWDOWN = 2.5


# =============================================================================
# 속도 노브의 CLI 진입점
# =============================================================================
# 궤적 CSV 를 내는 스크립트는 셋이다(glns/verify.py, trajectory/plan_move.py,
# trajectory/tilt_motion.py). 플래그를 각자 정의하면 이름과 기본값이 조용히 어긋나므로
# scene_config.add_cli_argument 와 같은 방식으로 한 곳에 둔다.
#
# (CLI 플래그, 이 모듈의 이름, 도움말)
TIMING_KNOBS = (
    ("--ee-speed-mm-s", "EE_SPEED_MM_S",
     "스캔 구간 EE 선속도 [mm/s]. 0 이면 이 제한을 끈다"),
    ("--ee-angular-speed-deg-s", "EE_ANGULAR_SPEED_DEG_S",
     "스캔 구간 EE 각속도 [deg/s]. 0 이면 이 제한을 끈다"),
    ("--max-joint-vel-rad-s", "MAX_JOINT_VEL_RAD_S",
     "관절 속도 상한 [rad/s]. 스캔과 transit 양쪽에 걸린다 — transit 시간은 이 값만으로 정해진다"),
    ("--min-segment-dt-s", "MIN_SEGMENT_DT_S",
     "segment 최소 소요시간 [s]. 짧은 구간이 순간적으로 빨라지는 것을 막는 바닥값"),
    ("--corner-angle-threshold-deg", "CORNER_ANGLE_THRESHOLD_DEG",
     "이 각도보다 급한 꺾임부터 감속을 시작한다 [deg]"),
    ("--corner-max-slowdown", "CORNER_MAX_SLOWDOWN",
     "가장 급한 꺾임에서의 감속 배수 (1.0 = 감속 없음)"),
)


def _non_negative(text: str) -> float:
    """argparse ``type=`` — 음수 속도/시간은 조용히 이상하게 동작하므로 여기서 막는다.

    0 은 막지 않는다: compute_trajectory_times 에서 "이 제한을 쓰지 않는다" 는 뜻이다.
    """
    import argparse

    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"could not parse as a number ({exc})")
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"must be >= 0 (got {value})")
    return value


def add_timing_cli_arguments(parser) -> None:
    """속도 노브 플래그를 parser 에 단다 — CSV 를 내는 스크립트는 전부 이걸 쓴다."""
    group = parser.add_argument_group(
        "timing (실행 속도)",
        "CSV 의 time 열을 정한다 = 실행 속도. 생성 시점에 궤적에 구워지므로 "
        "바꾸면 다시 생성해야 반영된다.")
    for flag, name, help_text in TIMING_KNOBS:
        group.add_argument(flag, type=_non_negative, default=None,
                           help=f"{help_text} (기본 {globals()[name]})")


def apply_timing_cli(args, *, verbose: bool = True) -> dict:
    """받은 override 를 이 모듈에 반영하고 **최종 값 전부**를 dict 로 돌려준다.

    반환값은 npz meta 로 간다(collision_gate_and_save 가 자동으로 넣는다) — 어떤 속도로
    구운 궤적인지는 CSV 를 봐서는 복원할 수 없는 정보다.
    """
    values, changed = {}, []
    for flag, name, _help in TIMING_KNOBS:
        override = getattr(args, flag.lstrip("-").replace("-", "_"), None)
        if override is not None and float(override) != globals()[name]:
            changed.append(f"{name} {globals()[name]} -> {float(override)}")
        if override is not None:
            globals()[name] = float(override)
        values[name] = globals()[name]
    if verbose:
        print(f"  Timing: EE {values['EE_SPEED_MM_S']:g} mm/s, "
              f"{values['EE_ANGULAR_SPEED_DEG_S']:g} deg/s, joint "
              f"{values['MAX_JOINT_VEL_RAD_S']:g} rad/s, min dt "
              f"{values['MIN_SEGMENT_DT_S']:g} s, corner "
              f">{values['CORNER_ANGLE_THRESHOLD_DEG']:g} deg x"
              f"{values['CORNER_MAX_SLOWDOWN']:g}")
        if changed:
            print("           overridden: " + "; ".join(changed))
    return values


def timing_values() -> dict:
    """지금 이 모듈에 걸려 있는 속도 노브 전부. CLI 를 안 쓰는 호출자용."""
    return {name: globals()[name] for _flag, name, _help in TIMING_KNOBS}


"""도면 없는 물체 복원 — 손목 depth 카메라로 찍어 표면을 다시 만든다.

라이브러리 (아래로 갈수록 무거운 의존):
  sensor.py        스캔 카메라 스펙·투영                         numpy 만
  storage.py       스캔 폴더 규약: 계획·프레임·복원 결과 저장/로드   numpy 만
  segment.py       depth 픽셀 판정 (물체 / ROI 밖 / 씬 장애물 / 무효)
  planning.py      ROI 둘레 스캔 자세 → IK → 자기 가림 제거 → ScanPlan   (cuRobo)
  registration.py  ICP 다중 정렬로 카메라 pose 보정                 (Open3D)
  pipeline.py      reconstruct(): 마스킹 → (ICP) → TSDF → 메시     (Open3D) — 앱의 진입점
  sim.py           시뮬레이션 전용: 정답 메시·ROI·pose 오차 주입·평가 — 실물 경로는 안 쓴다

CLI (단계마다 프로세스가 다르다 — Isaac 의 Python 과 cuRobo 를 섞지 않는 리포 규약):
  scan_plan.py                 (uv, cuRobo)  계획 → plan.npz
  ../isaac/scan_capture.py     (Isaac)       관절 순간이동 → depth 렌더 → frames/
  reconstruct.py               (uv, Open3D)  프레임 → recon.ply (+ 정답 대비 평가)
  export_recon.py              (uv)          recon.ply → 정답 물체 좌표의 recon.obj (평가용)
  ../../apps/scan_viewer.py    (viser)       점군·복원·정답 보기

여기서는 아무것도 import 하지 않는다 — Isaac 프로세스가 sensor·storage 만 가져가는데, 패키지
__init__ 이 planning(cuRobo)·pipeline(Open3D) 을 끌고 오면 안 되기 때문이다.
"""

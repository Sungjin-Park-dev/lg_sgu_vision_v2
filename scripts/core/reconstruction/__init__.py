"""도면 없는 물체 복원 — 손목 depth 카메라로 찍어 표면을 다시 만든다.

단계마다 프로세스가 다르다(Isaac 의 Python 과 cuRobo 를 섞지 않는 리포 규약):
  scan_plan.py          (uv, cuRobo)  관심 영역 둘레 스캔 자세 → IK → plan.npz
  core/isaac/scan_capture.py (Isaac)  관절 순간이동 → depth 렌더 → frame_*.npz
  view_scan.py          (uv)          프레임을 world 로 역투영해 viser 로 보고, 정답 메시와 거리 확인

sensor.py 는 numpy 만 쓴다 — 세 단계가 모두 import 한다.
"""

"""실물 로봇 스캔 — 손목 depth 카메라의 장착 보정과 촬영. 복원은 core.reconstruction 이 한다.

Isaac 촬영(core/isaac/scan_capture.py)의 실물판이다. 같은 저장 규약(data/{object}/scan/{run})으로
프레임을 남기고, 복원·ICP·TSDF 는 core.reconstruction.pipeline 을 그대로 쓴다.

  capture.py      강체 변환·depth 단위·관절 이력(정지 판정)·카메라 프레임·ScanSession   numpy 만
  calibration.py  ChArUco·hand-eye·별도 자세 검증·보드 depth 평면 검사                 (OpenCV)
  piper.py        AgileX PiPER 전용: joint1~6, URDF FK, CAN 수신 전용 SDK              (yourdfpy, pyAgxArm)
  sources.py      프레임 소스: ROS RealSense + PiPER / 데모 / 가상 보정                 (rclpy)

로봇에 명령을 보내지 않는다(PassivePiper 는 SDK 송신을 막는다). UR 셀 설정(common.config)도 읽지 않는다.
앱: apps/piper_scan.py (ros2-jazzy-piper 컨테이너의 .venv-piper — numpy 1.x). 여기와
core.reconstruction 의 코드는 메인 .venv(numpy 2.x)에서도 돌아야 한다(scripts/checks/check_piper_scan.py).

여기서는 아무것도 import 하지 않는다 — 하위 모듈마다 무거운 의존이 다르다.
"""

# AgileX PiPER

`piper_description.urdf` — [agx_arm_ros](https://github.com/agilexrobotics/agx_arm_ros) `e4ccec1`의
`src/agx_arm_description/agx_arm_urdf/piper/urdf/piper_description.urdf` 그대로 (MIT, `LICENSE`).

- **바이트 단위로 고치지 않는다.** 손목 보정(`workcell/calibration/piper_d405.yaml`)의 `identity.urdf_sha256`
  이 이 파일의 해시다. 주석 한 줄만 바꿔도 앱이 "보정 당시 URDF 와 다르다"며 재보정을 요구한다.
  sha256 `8c910f07d14652dc0c50e91f3b8834723e1a465855d2c7fe59f503a7655c3cf4`
- 메시(`package://agx_arm_description/...`)는 복사하지 않았다 — 스캔 앱은 FK 만 쓴다(`load_meshes=False`).
- 원본이 바뀌면 다시 복사하고 손목 보정을 새로 한다.

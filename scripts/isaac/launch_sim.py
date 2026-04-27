#!/usr/bin/env python3
"""
빈 Isaac Sim 시뮬레이터만 실행.

URDF/USD 등을 GUI에서 직접 import해서 테스트할 때 사용.

사용법:
    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/isaac/launch_sim.py
"""

import numpy as np
from isaacsim import SimulationApp

CONFIG = {"renderer": "RaytracedLighting", "headless": False}
simulation_app = SimulationApp(CONFIG)

from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import extensions, viewports

# Enable useful extensions for manual workflow
extensions.enable_extension("isaacsim.ros2.bridge")
extensions.enable_extension("isaacsim.asset.importer.urdf")

simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)
viewports.set_camera_view(eye=np.array([1.5, 1.5, 1.0]), target=np.array([0, 0, 0.5]))

simulation_app.update()

while simulation_app.is_running():
    simulation_context.step(render=True)

simulation_app.close()

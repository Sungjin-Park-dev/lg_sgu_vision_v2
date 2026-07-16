"""Trajectory CSV and sidecar persistence."""

from pathlib import Path

import numpy as np

def save_trajectory_csv(solutions, ee_positions, ee_quaternions, output_path,
                        robot_name="ur20", dt=1.0, times=None):
    """Trajectory를 CSV로 저장. joint 컬럼에 robot_name prefix 추가."""
    import csv
    import os
    import tempfile

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    header = ["time"] + [f"{robot_name}-{j}" for j in JOINT_NAMES] + [
        "target-POS_X", "target-POS_Y", "target-POS_Z",
        "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W",
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(solutions)):
                t = times[i] if times is not None else i * dt
                row = [float(t)] + solutions[i].tolist()
                row += ee_positions[i].tolist()
                row += ee_quaternions[i].tolist()
                writer.writerow(row)

        tmp_path.chmod(0o644)
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    print(f"  CSV saved to {output_path} ({len(solutions)} waypoints)")


# =========================================================================
# Main
# =========================================================================

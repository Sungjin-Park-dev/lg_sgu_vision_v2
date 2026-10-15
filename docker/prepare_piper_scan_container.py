#!/usr/bin/env python3
"""Host: preserve the existing PiPER container and add the vision repo bind mount.

No changes to CAN, motors or ROS nodes. Refuses migration while robot processes
are running. Keeps the original container and a committed image for rollback.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = "/workspace/lg_sgu_vision_v2"


def run(*args, capture=False):
    result = subprocess.run(["docker", *args], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container", default="ros2-jazzy-piper")
    args = p.parse_args()
    old = json.loads(run("inspect", args.container, capture=True))[0]
    mounts = old["Mounts"]
    for mount in mounts:
        if mount["Destination"] == DESTINATION:
            if mount["Type"] != "bind" or Path(mount["Source"]).resolve() != ROOT:
                raise RuntimeError("Project mount already points to another directory")
            if not old["State"]["Running"]:
                run("start", args.container)
            print(f"Already mounted: {ROOT} -> {DESTINATION}")
            return
    if old["State"]["Running"]:
        processes = run("exec", args.container, "ps", "-eo", "stat=,args=", capture=True)
        robot = re.compile(r"(?:ros2 launch|agx_arm_ctrl_single|move_group|rviz2|realsense2_camera_node|piper_scan.py)")
        if any(robot.search(line) and not line.strip().startswith("Z") for line in processes.splitlines()):
            raise RuntimeError("Stop PiPER driver/MoveIt/RViz/camera/scan processes with Ctrl+C before migration")
    config, hc = old["Config"], old["HostConfig"]
    if hc["NetworkMode"] != "host" or any(m["Type"] != "bind" for m in mounts):
        raise RuntimeError("Expected the inspected host-network, bind-mounted PiPER container")
    if hc.get("Privileged") or hc.get("CapAdd") or hc.get("CapDrop"):
        raise RuntimeError("Unexpected capability settings; migration refuses to change them")
    requests = hc.get("DeviceRequests") or []
    if requests and (len(requests) != 1 or requests[0]["Count"] != -1):
        raise RuntimeError("Unsupported GPU config; the existing container was left unchanged")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{args.container}-before-scan-{stamp}"
    image = f"ros2-jazzy-piper-scan-base:{stamp}"
    out = ROOT / "docker/.piper-backups" / stamp
    out.mkdir(parents=True, exist_ok=False)
    (out / "inspect.json").write_text(json.dumps(old, indent=2))
    run("commit", args.container, image)
    if old["State"]["Running"]:
        run("stop", "--time", "5", args.container)
    run("rename", args.container, backup)
    create = ["create", "--name", args.container, "--hostname", config["Hostname"],
              "--network", "host", "--ipc", hc["IpcMode"], "--shm-size", str(hc["ShmSize"]),
              "--restart", hc["RestartPolicy"]["Name"] or "no"]
    for value in config.get("Env") or []:
        create += ["--env", value]
    for mount in mounts:
        # Use --mount (not -v), so missing Xauth/devices cannot silently become directories.
        spec = f"type=bind,src={mount['Source']},dst={mount['Destination']}"
        if not mount["RW"]:
            spec += ",readonly"
        create += ["--mount", spec]
    create += ["--mount", f"type=bind,src={ROOT},dst={DESTINATION}"]
    for device in hc.get("Devices") or []:
        create += ["--device", f"{device['PathOnHost']}:{device['PathInContainer']}:{device['CgroupPermissions']}"]
    for rule in hc.get("DeviceCgroupRules") or []:
        create += ["--device-cgroup-rule", rule]
    if requests:
        create += ["--gpus", "all"]
    if config.get("WorkingDir"):
        create += ["--workdir", config["WorkingDir"]]
    if config.get("User"):
        create += ["--user", config["User"]]
    if config.get("Tty"):
        create += ["--tty"]
    if config.get("OpenStdin"):
        create += ["--interactive"]
    for key, value in (config.get("Labels") or {}).items():
        create += ["--label", f"{key}={value}"]
    if config.get("Entrypoint"):
        create += ["--entrypoint", config["Entrypoint"][0]]
    create += [image, *(config.get("Entrypoint") or [])[1:], *(config.get("Cmd") or [])]
    created = False
    try:
        run(*create)
        created = True
        run("start", args.container)
    except BaseException:
        if created:
            run("rm", "--force", args.container)  # Only the fresh, unused replacement.
        run("rename", backup, args.container)
        if old["State"]["Running"]:
            run("start", args.container)
        raise
    (out / "rollback.txt").write_text(
        f"Original container: {backup}\nCommitted image: {image}\n"
        f"Stop new container, rename it aside, then rename {backup} to {args.container}.\n")
    print(f"Ready: {args.container}; backup: {backup}; image: {image}")


if __name__ == "__main__":
    main()

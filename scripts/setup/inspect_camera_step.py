#!/usr/bin/env python3
"""Measure the camera CAD (STEP) along the optical axis, in the URDF flange frame.

This is where the numbers in docs/reference/camera-geometry.md come from. No CAD kernel
is installed (no OCP/pythonocc), so this reads the ISO-10303-21 text directly: it walks
the assembly tree, composes each occurrence's placement, and projects B-rep *vertices*
onto the optical axis. Vertices -- not raw CARTESIAN_POINTs -- because spline control
points and construction geometry sit off the solid and inflate the extents.

The flange origin is the robot-side face of the ZIVID adapter plate. The check that this
is the right datum: the lens barrel tip must land on build_camera_mesh.EXPECT_HI[0]
(0.21877 m), which is what the baked mesh actually measures.

Usage:
    uv run --no-sync scripts/setup/inspect_camera_step.py
    uv run --no-sync scripts/setup/inspect_camera_step.py --step <file.stp> --no-assert
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP = PROJECT_ROOT / "workcell" / "robot" / "camera" / "source" / "camera_asm_wo_light.stp"

FLANGE_DATUM_PART = "ZIVID_END_EFFECTOR_UR20_PT2"   # its rear face bolts to the UR flange
LENS_PART = "MFA121-U50"                            # optical axis comes from its cylinders

# Landmarks asserted against the rest of the repo (mm from flange, along the optical axis).
EXPECT = {
    "lens barrel tip": (218.770, "URDF camera_optical_joint xyz / TOOL_TO_CAMERA_OPTICAL_OFFSET_M"
                                 " / build_camera_mesh.EXPECT_HI[0] = 0.21877 m"),
    "body_face": (141.000, "커버 앞면. 2026-08-27 부터 코드 기준점이 아니다 — CAD 랜드마크로만 남는다"),
    "object_plane": (391.000, "VIEW_1 판 = 벤더 공칭 WD 250mm 의 물체면(body_face+250)."
                              " config 기본 WD 는 실사용 값이라 여기와 다르다"),
}
TOL_MM = 1e-3


def parse_entities(path: Path) -> dict:
    src = open(path, encoding="utf-8", errors="replace").read()
    src = src[src.index("DATA;"):]
    ent = {}
    for m in re.finditer(r"#(\d+)\s*=\s*(.*?);", src, re.S):
        body = m.group(2).strip()
        tm = re.match(r"([A-Z_0-9]+)\s*\((.*)\)\s*$", body, re.S)
        # A "complex" instance -- "( A(..) B(..) )" -- has no single type name.
        ent[int(m.group(1))] = (tm.group(1), tm.group(2)) if tm else ("_COMPLEX", body)
    return ent


class Step:
    def __init__(self, path: Path):
        self.ent = parse_entities(path)
        self._geo_cache: dict = {}

    # --- entity accessors ---------------------------------------------------
    def refs(self, arg: str):
        return [int(x) for x in re.findall(r"#(\d+)", arg)]

    def strs(self, arg: str):
        return re.findall(r"'([^']*)'", arg)

    def nums(self, arg: str):
        return [float(x.replace("E", "e"))
                for x in re.findall(r"-?\d+\.?\d*(?:[Ee][+-]?\d+)?", arg)]

    def of_type(self, t: str):
        return [(i, a) for i, (ty, a) in self.ent.items() if ty == t]

    def vec3(self, eid: int) -> np.ndarray:
        return np.array(self.nums(self.ent[eid][1])[:3], dtype=float)

    def placement(self, eid: int) -> np.ndarray:
        """AXIS2_PLACEMENT_3D -> 4x4."""
        r = self.refs(self.ent[eid][1])
        o = self.vec3(r[0])
        z = self.vec3(r[1]) if len(r) > 1 else np.array([0.0, 0.0, 1.0])
        x = self.vec3(r[2]) if len(r) > 2 else np.array([1.0, 0.0, 0.0])
        z = z / np.linalg.norm(z)
        x = x - z * (x @ z)
        x = x / np.linalg.norm(x)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
        T[:3, 3] = o
        return T

    # --- assembly structure -------------------------------------------------
    def occurrences(self):
        """[(name, trail, vertices_in_root_frame, cylinders)] for every leaf occurrence.

        `trail` is the '/'-joined ancestor chain -- sub-assemblies like MFA121-U50 carry no
        geometry of their own, so membership has to be tested against the path, not the name.
        """
        name_of = {}
        for i, a in self.of_type("PRODUCT_DEFINITION"):
            r = self.refs(a)                                   # [formation, context]
            prod = self.refs(self.ent[r[0]][1]) if r else []
            name_of[i] = self.strs(self.ent[prod[0]][1])[0] if prod else f"#{i}"

        shape_of = {i: self.refs(a)[-1]
                    for i, a in self.of_type("PRODUCT_DEFINITION_SHAPE") if self.refs(a)}
        rep_of = {}
        for i, a in self.of_type("SHAPE_DEFINITION_REPRESENTATION"):
            r = self.refs(a)
            if len(r) >= 2 and r[0] in shape_of:
                rep_of[shape_of[r[0]]] = r[1]
        usage = {i: (self.refs(a)[0], self.refs(a)[1])
                 for i, a in self.of_type("NEXT_ASSEMBLY_USAGE_OCCURRENCE")}

        kids: dict = {}
        for i, a in self.of_type("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"):
            r = self.refs(a)
            nid = self.refs(self.ent[r[1]][1])[-1]
            if nid not in usage:
                continue
            parent, child = usage[nid]
            rel = self.refs(self.ent[r[0]][1])
            idt = next((x for x in rel
                        if self.ent.get(x, ("",))[0] == "ITEM_DEFINED_TRANSFORMATION"), None)
            if idt is None:
                continue
            # REPRESENTATION_RELATIONSHIP(rep_1, rep_2) + a transform mapping rep_1 -> rep_2.
            # Creo writes rep_1 = child, rep_2 = parent; don't trust it, check.
            ax = self.refs(self.ent[idt][1])
            T = self.placement(ax[1]) @ np.linalg.inv(self.placement(ax[0]))
            if rep_of.get(child) == rel[1] and rep_of.get(parent) == rel[0]:
                T = np.linalg.inv(T)
            kids.setdefault(parent, []).append((child, T))

        out = []

        def walk(pd, T, trail):
            rep = rep_of.get(pd)
            if rep is not None:
                verts, cyls = self._geometry(rep)
                if len(verts):
                    out.append((name_of[pd], trail,
                                (T[:3, :3] @ verts.T).T + T[:3, 3],
                                [(T[:3, :3] @ d, r) for d, r in cyls]))
            for child, Tc in kids.get(pd, []):
                walk(child, T @ Tc, f"{trail}/{name_of[pd]}")

        embedded = {c for cs in kids.values() for c, _ in cs}
        for root in (pd for pd in name_of if pd not in embedded):
            walk(root, np.eye(4), "")
        return out

    def _geometry(self, rep: int):
        if rep in self._geo_cache:
            return self._geo_cache[rep]
        seen, stack, verts, cyls = set(), list(self.refs(self.ent[rep][1])), [], []
        while stack:
            e = stack.pop()
            if e in seen or e not in self.ent:
                continue
            seen.add(e)
            ty, a = self.ent[e]
            if ty == "VERTEX_POINT":
                r = self.refs(a)
                if r and self.ent[r[-1]][0] == "CARTESIAN_POINT":
                    v = self.nums(self.ent[r[-1]][1])
                    if len(v) >= 3:
                        verts.append(v[:3])
            elif ty == "CYLINDRICAL_SURFACE":
                r = self.refs(a)
                cyls.append((self.placement(r[0])[:3, 2], self.nums(a)[-1]))
            stack.extend(self.refs(a))
        self._geo_cache[rep] = (np.array(verts) if verts else np.zeros((0, 3)), cyls)
        return self._geo_cache[rep]


def optical_axis(parts) -> np.ndarray:
    """Dominant cylinder axis of the lens assembly, pointed away from the robot."""
    votes: dict = {}
    for name, trail, _, cyls in parts:
        if LENS_PART not in trail and LENS_PART not in name:
            continue
        for d, _r in cyls:
            d = d / np.linalg.norm(d)
            if d[np.argmax(np.abs(d))] < 0:
                d = -d
            votes[tuple(np.round(d, 6))] = votes.get(tuple(np.round(d, 6)), 0) + 1
    if not votes:
        raise SystemExit(f"no {LENS_PART} cylinders found -- is this the right STEP?")
    axis = np.array(max(votes.items(), key=lambda kv: kv[1])[0], dtype=float)

    lens = np.vstack([v for _n, t, v, _ in parts if LENS_PART in t])
    plate = np.vstack([v for n, _t, v, _ in parts if n.startswith(FLANGE_DATUM_PART)])
    return -axis if (lens @ axis).mean() < (plate @ axis).mean() else axis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=Path, default=DEFAULT_STEP)
    ap.add_argument("--no-assert", action="store_true",
                    help="print the table without checking the repo's landmark values")
    args = ap.parse_args()

    print(f"Reading {args.step.relative_to(PROJECT_ROOT)}")
    parts = Step(args.step).occurrences()
    axis = optical_axis(parts)

    plate = [v for n, _t, v, _ in parts if n.startswith(FLANGE_DATUM_PART)]
    if not plate:
        raise SystemExit(f"{FLANGE_DATUM_PART} not found -- cannot locate the flange datum")
    origin = min((v @ axis).min() for v in plate)

    print(f"  optical axis (assembly root): {np.round(axis, 6).tolist()}")
    print(f"  flange datum ({FLANGE_DATUM_PART} rear face) at root s={origin:.3f} mm\n")
    print(f"{'from':>9s} {'to':>9s} {'width':>8s}  part")

    rows = []
    for name, trail, verts, _ in parts:
        s = verts @ axis - origin
        perp = verts - np.outer(verts @ axis, axis)
        rows.append((s.min(), s.max(), float((perp.max(0) - perp.min(0)).max()),
                     name, trail))
    for lo, hi, w, name, _trail in sorted(rows):
        print(f"{lo:9.3f} {hi:9.3f} {w:8.2f}  {name}")

    measured = {
        "lens barrel tip": max(hi for _, hi, _, _n, t in rows if LENS_PART in t),
        # _X2_CE21BA74_X0_ is Creo's UTF-16 escape for 커버 (the cover); its front face is body_face.
        "body_face": max(hi for _, hi, _, n, _t in rows if n.startswith("_X2_CE21BA74")),
        "object_plane": min(lo for lo, _, _, n, _t in rows if n.startswith("VIEW_1")),
    }
    print()
    ok = True
    for key, value in measured.items():
        expected, why = EXPECT[key]
        hit = abs(value - expected) <= TOL_MM
        ok &= hit
        print(f"  {key:16s} {value:9.3f} mm   expected {expected:9.3f}  "
              f"{'OK' if hit else 'MISMATCH'}   ({why})")
    if not ok and not args.no_assert:
        raise SystemExit("\nCAD no longer matches the values the repo is built on -- "
                         "see docs/reference/camera-geometry.md before changing anything.")


if __name__ == "__main__":
    main()

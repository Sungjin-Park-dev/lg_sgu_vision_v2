#!/usr/bin/env python3
"""CAD(B-rep) 면 위에서 직접 viewpoint 를 뽑는 샘플러.

표면 FPS 는 **삼각형** 위에 점을 흩뿌린다. 여기서는 STEP 의 곡면 정의를 그대로 쓴다:

  1. 면마다 (u,v) 매개변수 평면에 FOV 간격 격자를 깐다 — 매개변수는 거리가 아니므로
     제1기본형식(``|∂S/∂u|``)으로 mm 로 환산한다(원통의 u 는 각도라 R 을 곱해야 한다).
  2. 격자점이 **트리밍 경계 안**인지 판정한다. STEP 의 ``PLANE`` 은 무한 평면이고 면은
     그 위에서 경계 루프로 오려낸 조각이라, 이 판정이 없으면 윤곽 밖과 구멍 한가운데에도
     점이 생긴다(= 로봇이 허공을 검사한다).
  3. 점과 법선을 곡면에서 해석적으로 평가한다 → **테셀레이션 오차가 없다**. 삼각형 기반은
     법선이 1° 틀리면 WD 195mm 에서 카메라가 3.4mm 어긋난다.

필렛은 기본적으로 건너뛴다. 면마다 격자를 깔면 ceil 반올림이 면 수만큼 쌓이는데, 필렛은
이웃 면의 촬영이 이미 덮기 때문이다(측정: square_structure 280→200점, 커버리지 99.5% 유지).
무엇이 필렛인지는 CAD 가 알려준다 — 원통/토러스 면의 반지름.

OCP(OpenCASCADE 파이썬 바인딩)가 필요하다. 163MB 라 선택적 의존성으로 두고, 없으면 이
모듈을 import 하는 시점이 아니라 **쓰는 시점**에 안내와 함께 실패한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from . import visibility
from .mesh import (ALIGN_AUTO, ALIGN_NONE, axis_align_transform,
                   bottom_center_transform, load_source_parts, up_axis_transform)

# 반지름이 이보다 작은 원통/토러스 면은 필렛으로 보고 건너뛴다(이웃 면이 덮는다).
DEFAULT_FILLET_MAX_RADIUS_MM = 8.0
# 이보다 작은 면은 격자를 깔아도 점 하나가 전부라 의미가 없다.
DEFAULT_MIN_FACE_AREA_CM2 = 0.05

_OCP_HINT = ("CAD 면 샘플러에는 OCP 가 필요하다 — `uv sync` (pyproject 의 cadquery-ocp)")


def _ocp():
    """OCP 심볼을 한 번에 가져온다. 없으면 무엇을 깔아야 하는지 말해준다."""
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepTools import BRepTools
        from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
        from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_SurfaceType, GeomAbs_Torus
        from OCP.GeomLProp import GeomLProp_SLProps
        from OCP.GProp import GProp_GProps
        from OCP.gp import gp_Pnt2d
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError(_OCP_HINT) from exc
    return {
        "BRep_Tool": BRep_Tool, "BRepAdaptor_Surface": BRepAdaptor_Surface,
        "BRepGProp": BRepGProp, "BRepTools": BRepTools,
        "BRepTopAdaptor_FClass2d": BRepTopAdaptor_FClass2d,
        "GeomAbs_Cylinder": GeomAbs_Cylinder, "GeomAbs_SurfaceType": GeomAbs_SurfaceType,
        "GeomAbs_Torus": GeomAbs_Torus, "GeomLProp_SLProps": GeomLProp_SLProps,
        "GProp_GProps": GProp_GProps, "gp_Pnt2d": gp_Pnt2d,
        "IFSelect_RetDone": IFSelect_RetDone, "STEPControl_Reader": STEPControl_Reader,
        "TopAbs_FACE": TopAbs_FACE, "TopAbs_IN": TopAbs_IN, "TopAbs_ON": TopAbs_ON,
        "TopAbs_REVERSED": TopAbs_REVERSED, "TopExp_Explorer": TopExp_Explorer,
        "TopoDS": TopoDS,
    }


def _xcaf_parts(step_path) -> dict:
    """부품 이름 → shape. 어셈블리면 컴포넌트별, 단일 솔리드면 최상위 하나.

    이름은 STEPCAFControl(XCAF) 로 읽는다 — 일반 STEPControl_Reader 는 형상만 주고 이름을
    버린다. cascadio 가 메시 쪽에 붙이는 이름과 같은 출처라 두 경로가 같은 키를 쓴다.
    """
    try:
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDataStd import TDataStd_Name
        from OCP.TDF import TDF_LabelSequence
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError(_OCP_HINT) from exc

    document = TDocStd_Document(TCollection_ExtendedString("viewpoint"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.ReadFile(str(step_path))
    reader.Transfer(document)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())

    def name_of(label, fallback):
        attribute = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
            return attribute.Get().ToExtString()
        return fallback

    parts, tops = {}, TDF_LabelSequence()
    tool.GetFreeShapes(tops)
    for i in range(1, tops.Length() + 1):
        label = tops.Value(i)
        components = TDF_LabelSequence()
        tool.GetComponents_s(label, components)
        if components.Length() == 0:
            parts[name_of(label, f"shape{i}")] = tool.GetShape_s(label)
            continue
        for j in range(1, components.Length() + 1):
            child = components.Value(j)
            parts[name_of(child, f"part{j}")] = tool.GetShape_s(child)
    return parts


def list_parts(step_path) -> List[str]:
    """STEP 안의 부품(솔리드) 이름들. 어셈블리가 아니면 길이 1."""
    return list(_xcaf_parts(step_path))


def read_faces(step_path, part_name: Optional[str] = None) -> Tuple[list, dict]:
    """STEP → (TopoDS_Face 목록, OCP 심볼). 면 순회 순서는 결정론적이다.

    ``part_name`` 을 주면 그 부품의 면만 돌려준다 — 어셈블리에서 지그를 빼고 검사 대상만
    샘플링할 때 쓴다.
    """
    o = _ocp()
    if part_name:
        parts = _xcaf_parts(step_path)
        shape = parts.get(part_name)
        if shape is None:
            lowered = {k.lower(): v for k, v in parts.items()}
            shape = lowered.get(str(part_name).lower())
        if shape is None:
            raise ValueError(f"part '{part_name}' not found in {Path(step_path).name} — "
                             f"available: {sorted(parts)}")
    else:
        reader = o["STEPControl_Reader"]()
        if reader.ReadFile(str(step_path)) != o["IFSelect_RetDone"]:
            raise ValueError(f"STEP 을 읽지 못했다: {step_path}")
        reader.TransferRoots()
        shape = reader.OneShape()
    faces, explorer = [], o["TopExp_Explorer"](shape, o["TopAbs_FACE"])
    while explorer.More():
        faces.append(o["TopoDS"].Face_s(explorer.Current()))
        explorer.Next()
    return faces, o


def _fillet_radius(face, o) -> Optional[float]:
    """원통/토러스면 그 반지름(mm) — 필렛 판정용. 아니면 None."""
    adaptor = o["BRepAdaptor_Surface"](face)
    kind = adaptor.GetType()
    if kind == o["GeomAbs_Cylinder"]:
        return float(adaptor.Cylinder().Radius())
    if kind == o["GeomAbs_Torus"]:
        return float(adaptor.Torus().MinorRadius())
    return None


def _surface_type_name(face, o) -> str:
    names = {getattr(o["GeomAbs_SurfaceType"], n): n.replace("GeomAbs_", "")
             for n in dir(o["GeomAbs_SurfaceType"]) if n.startswith("GeomAbs_")}
    return names.get(o["BRepAdaptor_Surface"](face).GetType(), "?")


def effective_fov_mm(radius_mm: float, fov_mm: float, wd_mm: float,
                     max_incidence_deg: float = 0.0, dof_mm: float = 0.0) -> float:
    """곡률이 있는 면에서 **실제로 쓸 수 있는** 프레임 폭(mm). 닫힌 형태로 풀린다.

    카메라는 중심점을 법선 방향에서 WD 만큼 떨어져 본다. 프레임 가장자리로 갈수록 표면이
    기울어 보이고(입사각) 카메라와의 거리가 달라진다(초점).

    입사각 한계는 **사인법칙**으로 풀린다. ``D = radius + WD`` (부호 있는 반지름) 라 두면

        D > 0 :  θ = α − asin( (radius / D)·sin α )
        D < 0 :  sin(θ + α) = (|R| / |D|)·sin α        (오목이고 |R| > WD 인 경우)

    한 식이 볼록·오목을 모두 덮는 이유는 오목에서 ``radius < 0`` 이라 asin 이 음수가 되어
    자동으로 더해지기 때문이다. 오목이 완만한 것도 여기서 나온다 — 법선이 도는 방향과
    시선이 도는 방향이 상쇄된다. ``|R| = WD`` 면 카메라가 곡률중심에 정확히 놓여 어디를 봐도
    입사각 0° 라 이 한계가 사라진다.

    심도는 다른 잣대를 쓴다. 초점면이 광축에 수직이라 초점을 정하는 것은 거리가 아니라
    **광축 방향 높이차(새그)** 이고, ``h = R(1 − cos θ) <= DoF`` 한 줄로 풀린다. 볼록·오목이
    같은 식을 쓴다.

    (오목면의 진짜 한계는 대개 주변 림의 가림과 카메라 몸체 간섭이다 — 그건 가시성 필터와
     cuRobo 충돌 검사가 따로 본다.)
    """
    if not np.isfinite(radius_mm) or radius_mm == 0.0:
        return fov_mm
    incidence_on = math.isfinite(max_incidence_deg) and max_incidence_deg > 0.0
    dof_on = math.isfinite(dof_mm) and dof_mm > 0.0
    if not incidence_on and not dof_on:
        return fov_mm
    convex = radius_mm > 0.0
    R = abs(float(radius_mm))
    D = R + float(wd_mm) if convex else float(wd_mm) - R
    # |D| ~ 0 은 카메라가 곡률중심에 놓인 경우다. 그때는 어디를 봐도 입사각이 0 이라 그 한계만
    # 사라진다 — 심도는 광축 방향 높이차(새그)로 재므로 **여전히 걸린다**.
    if abs(D) < 1e-9:
        incidence_on = False
    if not incidence_on and not dof_on:
        return fov_mm
    limits = [fov_mm]

    if incidence_on:
        # 사인법칙. 삼각형 C(곡률중심)-E(가장자리 점)-K(카메라) 에서 각의 합과
        # ``R / sin φ = |CK| / sin ψ`` 를 쓰면 θ 가 한 줄로 떨어진다.
        #
        #   D > 0 :  θ = α − asin( (R_signed / D)·sin α )
        #
        # **부호 있는 반지름**을 그대로 넣으면 볼록·오목이 한 식으로 합쳐진다 — 오목은
        # ``R_signed < 0`` 이라 asin 이 음수가 되어 자동으로 더해진다.
        #
        # ``D < 0`` 은 오목이면서 |R| > WD, 즉 카메라가 표면과 곡률중심 **사이**에 있는
        # 경우다. 그때만 곡률중심에서 본 각이 θ 가 아니라 180°−θ 로 잡혀 분기가 갈린다:
        #
        #   D < 0 :  sin(θ + α) = (|R| / |D|)·sin α
        #
        # (예전에는 코사인법칙을 제곱해 cos θ 에 대한 2차방정식으로 풀었다. 값은 같지만
        #  제곱 때문에 허근을 걸러내야 했다. 바꾸기 전에 두 가지로 검증했다 — 옛 2차식과
        #  140개 조합(볼록·오목, |R|≶WD, α 5~89°)에서 최대 2.2e-10mm 차이, 그리고 좌표를
        #  놓고 입사각을 직접 훑은 값과도 스윕 해상도 이내로 일치.)
        alpha = math.radians(max_incidence_deg)
        sin_a = math.sin(alpha)
        if D > 0.0:
            ratio = (float(radius_mm) / D) * sin_a
            if abs(ratio) <= 1.0:
                theta = alpha - math.asin(ratio)
                if theta > 0.0:
                    limits.append(2.0 * R * theta)
        else:
            ratio = (R / abs(D)) * sin_a
            if abs(ratio) <= 1.0:
                positive = [t for t in (math.asin(ratio) - alpha,
                                        math.pi - math.asin(ratio) - alpha) if t > 0.0]
                if positive:
                    limits.append(2.0 * R * min(positive))

    if dof_on:
        # 초점면은 광축에 수직이므로 초점을 정하는 것은 **광축 방향 높이차(새그)** 다.
        #     h = R·(1 − cos θ) <= DoF   →   cos θ = 1 − DoF/R
        # 볼록은 가장자리가 멀어지고 오목은 가까워지지만 |h| 는 같아 식이 하나로 통일된다.
        # 예전에는 카메라~표면점 **직선거리**로 쟀는데(코사인법칙), 두 가지가 문제였다:
        #   * 직선거리는 가장자리에서 시선이 비스듬해 새그보다 크게 나와 과하게 보수적(~5%)
        #   * 오목면에서 |R| > WD 이면 가장자리가 오히려 **멀어지는데** 식이 가까워진다고 보아
        #     cos θ 가 ±1 을 벗어나며 제한이 조용히 빠졌다(R=170·WD=75 에서 실측 확인).
        cos_theta = 1.0 - dof_mm / R
        if cos_theta > -1.0:               # -1 이하면 면 전체가 심도 안 — 제한 없음
            limits.append(2.0 * R * math.acos(min(1.0, cos_theta)))

    return max(1e-3, min(limits))


def directional_radii_mm(face, o, probes: int = 3) -> Tuple[float, float]:
    """프레임 두 축(u, v) **각각의** 부호 있는 곡률 반지름(mm). 평면 방향은 inf.

    곡률은 방향에 따라 다르다 — 원통은 둘레 방향으로만 휘고 축 방향은 완전히 평평하다.
    오일러 공식 ``κ(θ) = κ₁cos²θ + κ₂sin²θ`` 로 주곡률을 u/v 축에 투영한다. 축 하나에
    가장 급한 값을 몰아 쓰면(예전 방식) 평평한 방향까지 촘촘해져 점을 낭비한다
    (원통 옆면 108×80mm 기준 40칸 → 24칸).

    부호 규약은 ``effective_fov_mm`` 과 같다: 양수 = 볼록, 음수 = 오목.
    """
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Dir

    surface = o["BRep_Tool"].Surface_s(face)
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    sign = -1.0 if face.Orientation() == o["TopAbs_REVERSED"] else 1.0
    sharpest_u, sharpest_v = 0.0, 0.0
    for i in range(probes):
        for j in range(probes):
            u = u0 + (i + 0.5) * (u1 - u0) / probes
            v = v0 + (j + 0.5) * (v1 - v0) / probes
            props = o["GeomLProp_SLProps"](surface, float(u), float(v), 2, 1e-7)
            if not props.IsCurvatureDefined():
                continue
            d1, d2 = gp_Dir(), gp_Dir()
            props.CurvatureDirections(d1, d2)
            du = props.D1U()
            if du.Magnitude() < 1e-9:
                continue
            u_hat = np.array([du.X(), du.Y(), du.Z()]) / du.Magnitude()
            principal = np.array([d1.X(), d1.Y(), d1.Z()])
            cos2 = float(np.clip(abs(u_hat @ principal), 0.0, 1.0)) ** 2
            k1, k2 = float(props.MaxCurvature()), float(props.MinCurvature())
            k_u = -sign * (k1 * cos2 + k2 * (1.0 - cos2))       # 양수 = 볼록
            k_v = -sign * (k1 * (1.0 - cos2) + k2 * cos2)
            if abs(k_u) > abs(sharpest_u):
                sharpest_u = k_u
            if abs(k_v) > abs(sharpest_v):
                sharpest_v = k_v
    radius = lambda k: (1.0 / k) if abs(k) > 1e-9 else float("inf")
    return radius(sharpest_u), radius(sharpest_v)


def signed_radius_mm(face, o, probes: int = 3) -> float:
    """면에서 가장 급한 곡률 반지름(mm), **부호 포함**. 평면이면 inf.

    곡면 종류로 나누지 않고 주곡률을 직접 묻는다 — 평면은 0, 원통/구/토러스는 그 반지름,
    자유곡면(NURBS)은 (u,v)마다 다른 값이 그대로 나온다. 면 위 몇 점을 찍어 **곡률이 가장
    급한**(반지름이 가장 작은) 쪽을 쓴다 — 보수적이다.

    부호 규약: 바깥 법선 기준으로 볼록이면 양수, 오목이면 음수. effective_fov_mm 이 이
    부호로 곡률중심이 카메라 반대편인지 같은 쪽인지를 가른다.
    """
    from OCP.BRepTools import BRepTools
    surface = o["BRep_Tool"].Surface_s(face)
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    sign = -1.0 if face.Orientation() == o["TopAbs_REVERSED"] else 1.0
    sharpest = 0.0          # 부호 있는 곡률 중 절댓값이 가장 큰 것
    for i in range(probes):
        for j in range(probes):
            u = u0 + (i + 0.5) * (u1 - u0) / probes
            v = v0 + (j + 0.5) * (v1 - v0) / probes
            props = o["GeomLProp_SLProps"](surface, float(u), float(v), 2, 1e-7)
            if not props.IsCurvatureDefined():
                continue
            for kappa in (props.MaxCurvature(), props.MinCurvature()):
                signed = -sign * float(kappa)      # 양수 = 볼록
                if abs(signed) > abs(sharpest):
                    sharpest = signed
    return (1.0 / sharpest) if abs(sharpest) > 1e-9 else float("inf")


# 축 하나의 호길이를 적분할 때 쓰는 표본 수. 곡률이 급해도 65개면 0.1% 이내로 수렴한다.
ARC_SAMPLES = 65
# 프레임이 덮는 띠에서 계량을 몇 점 읽을지. **덮기 문제라 평균이 아니라 최댓값**을 쓴다 —
# 프레임 안 한 점이라도 간격이 FOV 를 넘으면 그게 구멍이다.
BAND_SAMPLES = 5


def _axis_positions(speed_at, t0: float, t1: float, fov_mm: float, overlap: float,
                    closed: bool) -> Tuple[np.ndarray, float]:
    """축 하나의 격자 좌표를 **호길이 기준**으로 배치한다. (매개변수 좌표, 실제 길이mm)

    매개변수는 거리가 아니고, 그 환산비(``|∂S/∂t|``)는 위치마다 다르다 — 구면의 경도 방향은
    ``R·cos(위도)`` 라 극으로 갈수록 좁아진다. 면 중점에서 한 번만 재서 상수로 쓰면 그만큼
    간격이 어긋나고, 큰 구면에서는 그게 커버리지 구멍으로 나타난다(측정: 298cm² 구면의 10%).
    그래서 축을 따라 속도를 적분해 누적 호길이를 만들고, **거리 기준으로** 점을 놓은 뒤
    매개변수로 되돌린다.

    간격은 ``fov×(1-overlap)`` 을 넘지 않는다. 열린 축은 양 끝에서 FOV/2 안쪽에 첫·마지막
    점을 두어 경계까지 덮고, 닫힌 축(원통 둘레)은 경계가 없으므로 균등 분할한다.
    """
    ts = np.linspace(t0, t1, ARC_SAMPLES)
    speed = np.array([max(float(speed_at(t)), 1e-12) for t in ts])
    arc = np.concatenate([[0.0], np.cumsum(np.diff(ts) * 0.5 * (speed[1:] + speed[:-1]))])
    length = float(arc[-1])
    if length <= 1e-9:
        return np.array([(t0 + t1) / 2.0]), 0.0
    step = max(fov_mm * (1.0 - overlap), 1e-6)
    if closed:
        count = max(1, math.ceil(length / step))
        targets = (np.arange(count) + 0.5) * length / count
    elif length <= fov_mm:
        targets = np.array([length / 2.0])              # 한 컷으로 덮인다
    else:
        count = max(2, math.ceil((length - fov_mm) / step) + 1)
        targets = np.linspace(fov_mm / 2.0, length - fov_mm / 2.0, count)
    return np.interp(targets, arc, ts), length


def sample_face(face, o, fov_w_mm: float, fov_h_mm: float, overlap: float,
                min_area_cm2: float = DEFAULT_MIN_FACE_AREA_CM2,
                working_distance_mm: float = 195.0,
                max_incidence_deg: float = 0.0, dof_mm: float = 0.0):
    """면 하나 → (points_mm, normals, extras, info). 격자 + 트리밍 + 해석적 법선·프레임축."""
    surface = o["BRep_Tool"].Surface_s(face)
    adaptor = o["BRepAdaptor_Surface"](face)
    u0, u1, v0, v1 = o["BRepTools"].UVBounds_s(face)
    props = o["GProp_GProps"]()
    o["BRepGProp"].SurfaceProperties_s(face, props)
    info = {"type": _surface_type_name(face, o), "area_cm2": props.Mass() / 100.0}
    if info["area_cm2"] < min_area_cm2:
        info["skipped"] = "tiny"
        return np.zeros((0, 3)), np.zeros((0, 3)), {}, info

    # 이 면에서 축별로 쓸 수 있는 프레임 폭. 곡률이 급한 축만 좁아진다 — 원통 축 방향은
    # 평평하므로 공칭 그대로 쓴다(예전에는 급한 쪽을 두 축에 몰아 써서 낭비했다).
    radius_u, radius_v = directional_radii_mm(face, o)
    fov_u = effective_fov_mm(radius_u, fov_w_mm, working_distance_mm,
                             max_incidence_deg, dof_mm)
    fov_v = effective_fov_mm(radius_v, fov_h_mm, working_distance_mm,
                             max_incidence_deg, dof_mm)
    info["radius_mm"] = (radius_u, radius_v)
    info["effective_fov_mm"] = (fov_u, fov_v)

    # 계량(제1기본형식)은 **2차원 장**이다. 한 줄에서만 읽으면 프레임이 두께를 갖는 순간
    # 어긋난다 — 구면의 첫 행은 띠 아래끝 0, 위끝 49.9mm/rad 로 속도가 두 배가 되어
    # 열 간격도 두 배로 벌어졌다(측정: FOV 50mm 자리에 78mm). 그래서 프레임이 덮는
    # 구간에서 여러 점을 읽고 **최댓값**을 쓴다.
    #
    # 해석적 곡면(평면·원통·원뿔·구·토러스)은 |∂S/∂u| 가 v 에만 의존하고 |∂S/∂v| 가
    # 상수라, 이 최댓값이 근사가 아니라 **정확한 최악값**이다. B-spline 면만 두 축 모두
    # 의존할 수 있어 BAND_SAMPLES 점 샘플링이 근사가 된다(보수적인 쪽으로).
    u_span = np.linspace(u0, u1, BAND_SAMPLES)

    def speeds_v(t):
        return [o["GeomLProp_SLProps"](surface, float(u), float(t), 1, 1e-7)
                .D1V().Magnitude() for u in u_span]

    # 행 배치는 최댓값으로 — 빨리 움직이는 곳 기준이라야 어디서도 FOV 를 안 넘는다.
    vs, len_v = _axis_positions(lambda t: max(speeds_v(t)), v0, v1, fov_v, overlap,
                                adaptor.IsVClosed())

    classifier = o["BRepTopAdaptor_FClass2d"](face, 1e-6)
    sign = -1.0 if face.Orientation() == o["TopAbs_REVERSED"] else 1.0
    inside_states = (o["TopAbs_IN"], o["TopAbs_ON"])
    points, normals, frames_u, frames_v = [], [], [], []
    n_u, len_u, grid_cells = 0, 0.0, 0
    for v in vs:
        # 이 행의 프레임이 v 로 덮는 매개변수 구간. 폭은 **가장 느린** 곳 기준이라야
        # 띠를 좁게 잡지 않는다(속도가 작을수록 같은 mm 가 더 넓은 Δv 다).
        half_v = (fov_v / 2.0) / max(min(speeds_v(v)), 1e-9)
        v_band = np.linspace(max(v0, v - half_v), min(v1, v + half_v), BAND_SAMPLES)
        us, len_u = _axis_positions(
            lambda t, _b=v_band: max(
                o["GeomLProp_SLProps"](surface, float(t), float(bv), 1, 1e-7)
                .D1U().Magnitude() for bv in _b),
            u0, u1, fov_u, overlap, adaptor.IsUClosed())
        n_u = max(n_u, len(us))
        grid_cells += len(us)
        for u in us:
            if classifier.Perform(o["gp_Pnt2d"](float(u), float(v))) not in inside_states:
                continue
            local = o["GeomLProp_SLProps"](surface, float(u), float(v), 1, 1e-7)
            if not local.IsNormalDefined():
                continue
            point, normal = local.Value(), local.Normal()
            unit_n = sign * np.array([normal.X(), normal.Y(), normal.Z()])
            # 촬영 프레임의 두 축. u 는 ∂S/∂u 방향, v 는 법선과의 외적으로 직교화한다 —
            # 우리 곡면(평면·원통·구·토러스)은 매개변수가 직교라 ∂S/∂v 와 일치한다.
            d1u = local.D1U()
            axis_u = np.array([d1u.X(), d1u.Y(), d1u.Z()])
            norm_u = np.linalg.norm(axis_u)
            if norm_u < 1e-9:
                continue
            axis_u = axis_u / norm_u
            axis_v = np.cross(unit_n, axis_u)
            norm_v = np.linalg.norm(axis_v)
            if norm_v < 1e-9:
                continue
            points.append([point.X(), point.Y(), point.Z()])
            normals.append(unit_n.tolist())
            frames_u.append(axis_u.tolist())
            frames_v.append((axis_v / norm_v).tolist())
    info.update(grid=(n_u, len(vs)), size_mm=(len_u, len_v), kept=len(points),
                dropped=grid_cells - len(points))
    extras = {
        "frame_u": np.asarray(frames_u, dtype=np.float64).reshape(-1, 3),
        "frame_v": np.asarray(frames_v, dtype=np.float64).reshape(-1, 3),
        "fov_u_mm": np.full(len(points), fov_u),
        "fov_v_mm": np.full(len(points), fov_v),
    }
    return (np.asarray(points, dtype=np.float64).reshape(-1, 3),
            np.asarray(normals, dtype=np.float64).reshape(-1, 3), extras, info)


def sample_step(step_path, fov_w_mm: float, fov_h_mm: float, overlap: float,
                fillet_max_radius_mm: float = DEFAULT_FILLET_MAX_RADIUS_MM,
                min_area_cm2: float = DEFAULT_MIN_FACE_AREA_CM2,
                part_name: Optional[str] = None, working_distance_mm: float = 195.0,
                max_incidence_deg: float = 0.0, dof_mm: float = 0.0,
                verbose: bool = True):
    """STEP(또는 그 안의 한 부품) → (points_mm, normals, meta). 좌표계는 CAD 그대로.

    ``meta`` 는 점별 배열을 담는다 — face_id(어느 면에서 왔나), 프레임 축 두 개, 축별 유효
    FOV. 프레임 축이 있어야 판정이 촬영 사각형을 사각형으로 볼 수 있다(내접원 근사는 간격
    = FOV 에서 대각선 틈을 만든다).
    """
    faces, o = read_faces(step_path, part_name)
    columns = {"face_id": [], "frame_u": [], "frame_v": [], "fov_u_mm": [], "fov_v_mm": []}
    all_points, all_normals, infos = [], [], []
    for index, face in enumerate(faces):
        radius = _fillet_radius(face, o)
        if fillet_max_radius_mm > 0 and radius is not None and radius < fillet_max_radius_mm:
            infos.append({"type": _surface_type_name(face, o), "skipped": "fillet",
                          "radius_mm": radius})
            continue
        points, normals, extras, info = sample_face(
            face, o, fov_w_mm, fov_h_mm, overlap, min_area_cm2,
            working_distance_mm=working_distance_mm,
            max_incidence_deg=max_incidence_deg, dof_mm=dof_mm)
        infos.append(info)
        if len(points):
            all_points.append(points)
            all_normals.append(normals)
            columns["face_id"].append(np.full(len(points), index, dtype=np.int32))
            for key in ("frame_u", "frame_v", "fov_u_mm", "fov_v_mm"):
                columns[key].append(extras[key])
    points = np.vstack(all_points) if all_points else np.zeros((0, 3))
    normals = np.vstack(all_normals) if all_normals else np.zeros((0, 3))
    meta = {"infos": infos}
    for key, chunks in columns.items():
        if not chunks:
            meta[key] = np.zeros((0, 3)) if key.startswith("frame") else np.zeros(0)
        else:
            meta[key] = np.vstack(chunks) if key.startswith("frame") \
                else np.concatenate(chunks)
    # 원 근사로 떨어질 때(프레임 축이 없는 경로)와 보고용 대표값. 두 축 중 큰 쪽을 쓴다.
    meta["effective_fov_mm"] = np.maximum(meta["fov_u_mm"], meta["fov_v_mm"])
    if verbose:
        sampled = sum(1 for i in infos if "skipped" not in i)
        fillets = sum(1 for i in infos if i.get("skipped") == "fillet")
        dropped = sum(i.get("dropped", 0) for i in infos)
        print(f"  CAD faces: {len(faces)} (sampled {sampled}, fillet-skipped {fillets}, "
              f"tiny {len(infos) - sampled - fillets})")
        print(f"  Grid points: {len(points)} kept, {dropped} outside trimming boundary")
        nominal = (fov_w_mm, fov_h_mm)
        narrowed = [i for i in infos
                    if any(e < n - 1e-6 for e, n in
                           zip(i.get("effective_fov_mm", nominal), nominal))]
        if narrowed:
            worst = min(min(i["effective_fov_mm"]) for i in narrowed)
            print(f"  Curvature-limited FOV on {len(narrowed)} face(s): "
                  f"down to {worst:.1f}mm (nominal {fov_w_mm:.0f}×{fov_h_mm:.0f}mm)")
    return points, normals, meta


def cad_to_reference_transform(step_path, reference_mesh,
                               tol_linear: Optional[float] = None,
                               align: str = ALIGN_AUTO,
                               part_name: Optional[str] = None) -> Tuple[np.ndarray, float]:
    """CAD 좌표(mm) → 파이프라인 좌표(m, 자세 정렬 + 바닥중심) 4x4 와 일치도.

    ``load_meshes`` 와 **같은 규칙**을 써야 한다 — 화면의 메시와 해석적 점이 다른 변환을
    타면 점이 물체에서 떠 보인다. 그래서 부품 필터도 자세 계산 **전에** 적용한다.
    """
    named = load_source_parts(step_path, tol_linear=tol_linear)
    if part_name:
        chosen = {k: v for k, v in named.items()
                  if k == part_name or k.lower() == str(part_name).lower()}
        if chosen:
            named = chosen
    parts = list(named.values())
    tess = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    if align == ALIGN_NONE:
        return np.eye(4), 1.0
    if align == ALIGN_AUTO:
        if reference_mesh is None:
            return bottom_center_transform(tess), 0.0
        rotate, score = axis_align_transform(tess, reference_mesh)
    else:
        rotate, score = up_axis_transform(align), 1.0
    tess = tess.copy()
    tess.apply_transform(rotate)
    return bottom_center_transform(tess) @ rotate, score


def sample_step_aligned(step_path, reference_mesh, fov_w_mm: float, fov_h_mm: float,
                        overlap: float,
                        fillet_max_radius_mm: float = DEFAULT_FILLET_MAX_RADIUS_MM,
                        tol_linear: Optional[float] = None, align: str = ALIGN_AUTO,
                        part_name: Optional[str] = None, working_distance_mm: float = 195.0,
                        max_incidence_deg: float = 0.0, dof_mm: float = 0.0,
                        verbose: bool = True):
    """``sample_step`` 결과를 파이프라인 좌표계(미터)로 옮겨 돌려준다.

    법선과 **프레임 축**도 같은 회전을 받는다 — 판정이 촬영 사각형을 이 축으로 재기 때문에
    하나라도 빠뜨리면 프레임이 표면 위에서 돌아간 채 계산된다.
    """
    points_mm, normals, meta = sample_step(
        step_path, fov_w_mm, fov_h_mm, overlap, fillet_max_radius_mm,
        part_name=part_name, working_distance_mm=working_distance_mm,
        max_incidence_deg=max_incidence_deg, dof_mm=dof_mm, verbose=verbose)
    transform, score = cad_to_reference_transform(
        step_path, reference_mesh, tol_linear, align=align, part_name=part_name)
    rotation = transform[:3, :3]
    points = (rotation @ (points_mm.T / 1000.0)).T + transform[:3, 3]

    def rotate_unit(vectors):
        out = (rotation @ np.asarray(vectors).reshape(-1, 3).T).T
        lengths = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(lengths < 1e-9, 1.0, lengths)

    normals = rotate_unit(normals)
    for key in ("frame_u", "frame_v"):
        if len(meta.get(key, ())):
            meta[key] = rotate_unit(meta[key])
    if verbose:
        print(f"  Aligned CAD → pipeline frame ({align}, match {score:.3f})")
    meta["transform"] = transform
    return points, normals, meta


# ============================================================================
# 커버리지 검증
# ============================================================================

DEFAULT_COVERAGE_CELL_MM = 4.0


@dataclass(frozen=True)
class SurfaceCells:
    """검사 대상 표면을 (u,v) 로 잘게 나눈 조각들 — **덮어야 할 원소**.

    커버리지 평가와 set covering 이 같은 것을 센다. 그래서 한 번 만들어 둘이 나눠 쓴다.
    좌표는 파이프라인 프레임(미터), 면적은 cm² 다.
    """

    points: np.ndarray          # (M,3)
    normals: np.ndarray         # (M,3)
    areas_cm2: np.ndarray       # (M,)
    face_id: np.ndarray         # (M,)
    transform: np.ndarray       # CAD(mm) → 파이프라인(m) 4x4
    # 셀의 (u,v) 접선. 셀을 겨냥한 **보충 viewpoint** 가 촬영 사각형의 방향으로 쓴다
    # (patch_uncovered). 격자 viewpoint 와 같은 규약이라 판정이 일관된다.
    frame_u: np.ndarray = None   # (M,3)
    frame_v: np.ndarray = None   # (M,3)

    def __len__(self) -> int:
        return len(self.points)


def surface_cells(step_path, *, part_name: Optional[str] = None,
                  tol_linear: Optional[float] = None, align: str = ALIGN_AUTO,
                  reference_mesh=None,
                  cell_mm: float = DEFAULT_COVERAGE_CELL_MM) -> SurfaceCells:
    """CAD 면들을 (u,v) 격자로 잘게 나눠 셀 목록을 만든다.

    표면을 무작위로 흩뿌려 세는 대신 **매개변수 공간을 나눈다**. 셀 면적을 제1기본형식
    ``|∂S/∂u × ∂S/∂v|·Δu·Δv`` 로 정확히 구할 수 있어 결과가 면적 가중이고, 어느 (u,v) 가
    비었는지도 남는다.
    """
    transform, _ = cad_to_reference_transform(
        step_path, reference_mesh, tol_linear, align=align, part_name=part_name)
    faces, o = read_faces(step_path, part_name)
    from OCP.BRepTools import BRepTools

    points, normals, areas, face_of, tan_u, tan_v = [], [], [], [], [], []
    for index, face in enumerate(faces):
        surface = o["BRep_Tool"].Surface_s(face)
        u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
        props = o["GProp_GProps"]()
        o["BRepGProp"].SurfaceProperties_s(face, props)
        if props.Mass() / 100.0 < 1e-3:
            continue
        mid = o["GeomLProp_SLProps"](surface, (u0 + u1) / 2.0, (v0 + v1) / 2.0, 1, 1e-7)
        du = max(mid.D1U().Magnitude(), 1e-9)
        dv = max(mid.D1V().Magnitude(), 1e-9)
        n_u = max(1, min(400, int(np.ceil((u1 - u0) * du / cell_mm))))
        n_v = max(1, min(400, int(np.ceil((v1 - v0) * dv / cell_mm))))
        classifier = o["BRepTopAdaptor_FClass2d"](face, 1e-6)
        sign = -1.0 if face.Orientation() == o["TopAbs_REVERSED"] else 1.0
        inside = (o["TopAbs_IN"], o["TopAbs_ON"])
        for i in range(n_u):
            for j in range(n_v):
                u = u0 + (i + 0.5) * (u1 - u0) / n_u
                v = v0 + (j + 0.5) * (v1 - v0) / n_v
                if classifier.Perform(o["gp_Pnt2d"](float(u), float(v))) not in inside:
                    continue
                local = o["GeomLProp_SLProps"](surface, float(u), float(v), 1, 1e-7)
                if not local.IsNormalDefined():
                    continue
                point, normal = local.Value(), local.Normal()
                d1u = np.array([local.D1U().X(), local.D1U().Y(), local.D1U().Z()])
                norm_u = np.linalg.norm(d1u)
                if norm_u < 1e-9:
                    continue
                unit_n = sign * np.array([normal.X(), normal.Y(), normal.Z()])
                axis_u = d1u / norm_u
                axis_v = np.cross(unit_n, axis_u)
                norm_v = np.linalg.norm(axis_v)
                if norm_v < 1e-9:
                    continue
                jac = np.linalg.norm(np.cross(
                    d1u, [local.D1V().X(), local.D1V().Y(), local.D1V().Z()]))
                areas.append(jac * ((u1 - u0) / n_u) * ((v1 - v0) / n_v))
                points.append([point.X(), point.Y(), point.Z()])
                normals.append(unit_n.tolist())
                tan_u.append(axis_u.tolist())
                tan_v.append((axis_v / norm_v).tolist())
                face_of.append(index)
    if not points:
        empty = np.zeros((0, 3))
        return SurfaceCells(empty, empty, np.zeros(0), np.zeros(0, dtype=np.int32), transform,
                            frame_u=empty, frame_v=empty)

    rot, offset = transform[:3, :3], transform[:3, 3]
    P = (rot @ (np.asarray(points).T / 1000.0)).T + offset       # CAD(mm) → 파이프라인(m)
    N = (rot @ np.asarray(normals).T).T
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    # 접선은 방향이라 회전만 씌운다(평행이동·스케일 없음).
    U = (rot @ np.asarray(tan_u).T).T
    V = (rot @ np.asarray(tan_v).T).T
    return SurfaceCells(points=P, normals=N,
                        areas_cm2=np.asarray(areas) / 100.0,     # mm² → cm²
                        face_id=np.asarray(face_of, dtype=np.int32),
                        transform=transform,
                        frame_u=U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12),
                        frame_v=V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12))


def inspectable(cells: SurfaceCells, bottom_angle_deg: float = 0.0,
                rotation=None, occluder=None,
                spec: Optional["visibility.SensorSpec"] = None) -> np.ndarray:
    """검사 **대상**인 셀 = True. 물리적으로 못 보는 셀은 분모에서 뺀다.

    알고리즘의 실패("구멍")와 물리적 불가("애초에 못 보는 곳")를 섞지 않기 위한 구분이다.
    두 가지를 뺀다:

      1. **아래를 향한 셀** — 로봇이 밑에서 올려다볼 수 없다. bottom filter 와 같은 규칙
         (visibility.facing_up).
      2. **가려진 셀** (``occluder`` 를 줬을 때) — 그 셀의 *이상적* 카메라, 즉 법선 위
         WD 지점에 놓은 카메라조차 가려서 못 보는 곳. 파인 홈 바닥, 지그 뒤, 속 빈 물체의
         안쪽 면이 여기 걸린다. 이런 셀을 분모에 남겨두면 커버리지가 "알고리즘이 못 덮었다"
         고 말하지만 실제로는 **어떤 알고리즘도 못 덮는다** — 실제로 square_structure 는
         이 때문에 67.1% 로 보고됐고, 모자란 255cm² 전부가 한 면(가려진 안쪽 판)이었다.

    ⚠️ 판정은 우리 후보 모형(법선 정면, WD 고정)을 따른다. 기울인 카메라라면 볼 수 있는
    셀도 '접근 불가' 로 빠진다 — 후보에 tilt 를 넣게 되면 이 기준도 같이 넓혀야 한다.
    """
    keep = visibility.facing_up(cells.normals, rotation, bottom_angle_deg)
    if occluder is not None and spec is not None and len(cells):
        keep &= visibility.self_visible(cells.points, cells.normals, occluder, spec)
    return keep


def inspection_mask(cells: SurfaceCells, positions, normals, point_fov_mm,
                    occluder, spec: "visibility.SensorSpec", *,
                    bottom_angle_deg: float = 0.0, rotation=None,
                    frames: Optional["visibility.ViewFrames"] = None):
    """커버리지의 **분모**가 될 셀 = True. ``(mask, unreachable_cm2)``.

    선택(set cover)과 커버리지 보고가 **같은 분모**를 봐야 한다. 아니면 greedy 가 더 작은
    과녁을 맞히고도 100% 를 주장한다 — 실제로 square_structure 에서 greedy 가 안쪽 벽을
    아예 목표에서 빼고 520cm² 에 100% 라고 보고했다(전부 사용은 632cm² 에 100%).
    그래서 이 마스크는 **후보 전체**로 한 번 계산해 둘이 나눠 쓴다.

    셀이 분모에 남는 조건은 둘 중 하나다:
      * 그 셀의 이상적 카메라(법선 위 WD)로 보인다 — 원리적으로 검사 가능
      * 실제로 어떤 후보가 그 셀을 검사한다 — 증거가 있으니 이상적 판정보다 강하다
    (뒤 조건이 없으면 비스듬히 들여다보이는 셀에서 분모가 분자보다 작아져 100% 를 넘는다.)
    """
    target = inspectable(cells, bottom_angle_deg, rotation)
    covered = visibility.covered_by_any(
        cells.points, cells.normals,
        np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        np.asarray(normals, dtype=np.float64).reshape(-1, 3),
        point_fov_mm, occluder, spec, mask=target, frames=frames)
    unreachable_cm2 = 0.0
    if occluder is not None and len(cells):
        reachable = visibility.self_visible(cells.points, cells.normals, occluder, spec)
        drop = target & ~reachable & ~covered
        unreachable_cm2 = float(cells.areas_cm2[drop].sum())
        target = target & ~drop
    return target, unreachable_cm2


def hole_candidates(cells: SurfaceCells, hole_mask, face_fov_mm: dict,
                    occluder, spec: "visibility.SensorSpec",
                    default_fov_mm: Tuple[float, float] = (50.0, 50.0),
                    verbose: bool = True):
    """구멍으로 남은 셀을 **정면으로 겨냥한** 후보 viewpoint. (points_m, normals, extras)

    격자를 건드리지 않고 후보 풀만 넓힌다. 격자를 손보는 대안(예: 트리밍 경계 밖 프레임을
    살리기)은 특정 원인 하나에만 듣고 격자의 규칙성을 깨뜨리는데, 이쪽은 **원인과 무관하게**
    남은 구멍에 듣는다 — 트리밍 경계든, 곡률이 급해 유효 FOV 가 좁아진 자리든.

    구멍 셀 하나당 후보 하나(그 셀의 법선 위 WD 지점에서 정면으로 본다)를 그대로 내놓는다.
    **여기서 고르지 않는다** — 줄이는 것은 select 한 곳에서만 한다. 예전에는 이 함수가
    자체 greedy 를 돌려 최소 집합만 내놨는데, 그러면 set cover 가 두 번 일어나 Selection 이
    무엇을 결정하는지 알 수 없었다.

    후보 적격성만 본다: 그 자리 카메라가 제 점을 볼 수 있어야 한다(가림) — 격자와 같은 규칙.

    Args:
        hole_mask: (M,) bool — 덮어야 하는데 안 덮인 셀.
        face_fov_mm: {face_id: (fov_u, fov_v)} — 면별 유효 FOV(샘플러의 ``infos``).
    """
    hole = np.asarray(hole_mask, dtype=bool)
    idx = np.flatnonzero(hole)
    empty = (np.zeros((0, 3)), np.zeros((0, 3)), {})
    if not len(idx) or cells.frame_u is None:
        return empty

    points, normals = cells.points[idx], cells.normals[idx]
    fov = np.array([face_fov_mm.get(int(f), default_fov_mm) for f in cells.face_id[idx]],
                   dtype=np.float64)
    ok = visibility.self_visible(points, normals, occluder, spec)
    if not ok.any():
        if verbose:
            print(f"  Hole candidates: {len(idx)} uncovered cells, none reachable")
        return empty
    keep = np.flatnonzero(ok)
    if verbose:
        print(f"  Hole candidates: {len(keep)} added for {int(hole.sum())} uncovered cells "
              f"({float(cells.areas_cm2[hole].sum()):.1f} cm²)")
    extras = {
        "face_id": cells.face_id[idx][keep],
        "frame_u": cells.frame_u[idx][keep], "frame_v": cells.frame_v[idx][keep],
        "fov_u_mm": fov[keep, 0], "fov_v_mm": fov[keep, 1],
        "effective_fov_mm": np.maximum(fov[keep, 0], fov[keep, 1]),
    }
    return points[keep], normals[keep], extras


def coverage_report(step_path, positions, normals, point_fov_mm, *,
                    working_distance_mm: float, occluder_mesh=None,
                    max_incidence_deg: float = 0.0, dof_mm: float = 0.0,
                    bottom_angle_deg: float = 0.0, rotation=None,
                    part_name: Optional[str] = None, tol_linear: Optional[float] = None,
                    align: str = ALIGN_AUTO, reference_mesh=None,
                    cell_mm: float = DEFAULT_COVERAGE_CELL_MM,
                    cells: Optional[SurfaceCells] = None,
                    frames: Optional["visibility.ViewFrames"] = None,
                    mask=None, unreachable_cm2: float = 0.0,
                    verbose: bool = True) -> dict:
    """생성된 viewpoint 가 각 CAD 면을 얼마나 덮는지 면적 가중으로 센다.

    셀은 셋 중 하나다:
      * **covered**  — 어떤 viewpoint 가 검사 가능한 조건으로 본다
      * **검사 불가** — 아래를 향하거나, 이상적 카메라로도 가려서 안 보임. 분모에서 뺀다
        (뺀 면적은 ``unreachable_cm2``)
      * **uncovered** — 덮을 수 있는데 안 덮인 곳. **이것만이 진짜 구멍이다**

    판정은 ``visibility`` 가 전담한다 — 필터·선택과 같은 코드를 쓰므로 셋이 어긋날 수 없다.
    ``frames`` 를 주면 촬영 프레임을 **사각형**으로 판정한다(없으면 내접원 근사).
    ``cells`` 를 주면 다시 만들지 않는다(선택 단계와 공유용).
    """
    if cells is None:
        cells = surface_cells(step_path, part_name=part_name, tol_linear=tol_linear,
                              align=align, reference_mesh=reference_mesh, cell_mm=cell_mm)
    if len(cells) == 0:
        return {"faces": {}, "covered_ratio": 0.0, "cells": 0,
                "covered_cm2": 0.0, "target_cm2": 0.0}

    spec = visibility.SensorSpec(
        working_distance_mm=working_distance_mm,
        max_incidence_deg=max_incidence_deg,
        depth_of_field_mm=dof_mm)
    # 분모는 **후보 전체**로 미리 정해 두고 넘겨받는다(mask). 선택 결과로 다시 계산하면
    # greedy 가 버린 셀이 분모에서도 사라져, 더 작은 과녁을 맞히고 100% 를 주장하게 된다.
    unreachable_cm2 = float(unreachable_cm2 or 0.0)
    if mask is None:
        target, unreachable_cm2 = inspection_mask(
            cells, positions, normals, point_fov_mm, occluder_mesh, spec,
            bottom_angle_deg=bottom_angle_deg, rotation=rotation, frames=frames)
    else:
        target = np.asarray(mask, dtype=bool)
    covered = visibility.covered_by_any(
        cells.points, cells.normals,
        np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        np.asarray(normals, dtype=np.float64).reshape(-1, 3),
        point_fov_mm, occluder_mesh, spec, mask=target, frames=frames)

    # 화면에 그릴 셀 상태: 0 = 구멍, 1 = 덮임, 2 = 검사 불가(아래 향함 또는 접근 불가).
    # 숫자만 주면 "어디가" 빠졌는지 알 수 없어 매번 진단 스크립트를 따로 짜야 했다.
    state = np.full(len(cells), 2, dtype=np.uint8)
    state[target & covered] = 1
    state[target & ~covered] = 0

    area, face_of = cells.areas_cm2, cells.face_id
    report, total_t, total_c = {}, 0.0, 0.0
    for index in sorted(set(face_of.tolist())):
        m = face_of == index
        target_cm2 = area[m & target].sum()
        covered_cm2 = area[m & target & covered].sum()
        report[int(index)] = {"area_cm2": float(area[m].sum()),
                              "target_cm2": float(target_cm2),
                              "covered_cm2": float(covered_cm2),
                              "ratio": float(covered_cm2 / target_cm2)
                              if target_cm2 > 1e-9 else 1.0}
        total_t += target_cm2
        total_c += covered_cm2
    result = {"faces": report, "covered_cm2": total_c, "target_cm2": total_t,
              "covered_ratio": (total_c / total_t) if total_t > 1e-9 else 1.0,
              "unreachable_cm2": unreachable_cm2, "cells": int(len(cells)),
              "cell_points": cells.points, "cell_state": state}
    if verbose:
        print(f"  Coverage: {result['covered_ratio']*100:.1f}% "
              f"({total_c:.1f}/{total_t:.1f} cm² of inspectable area, "
              f"{result['cells']} cells)")
        if unreachable_cm2 >= 0.05:
            print(f"    excluded {unreachable_cm2:.1f} cm² no camera can reach "
                  f"(occluded even from the ideal pose)")
        for index, row in sorted(report.items(), key=lambda kv: -kv[1]["target_cm2"])[:6]:
            if row["target_cm2"] < 0.5:
                continue
            print(f"    face {index:3d}: {row['ratio']*100:5.1f}% "
                  f"({row['covered_cm2']:6.1f}/{row['target_cm2']:6.1f} cm²)")
    return result

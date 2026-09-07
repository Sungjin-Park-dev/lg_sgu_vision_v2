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

_OCP_HINT = ("CAD 면 샘플러에는 OCP 가 필요하다 — `uv pip install cadquery-ocp` "
             "(또는 pyproject 의 optional-dependency 'cad')")


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
    기울어 보이고(입사각) 카메라와의 거리가 달라진다(초점). 중심에서 호길이 s = |R|·θ 인
    지점에서 두 값은 아래와 같고, 입사각 식을 cosθ 에 대해 정리하면 **2차방정식**이 된다.

        볼록(radius > 0, 곡률중심이 카메라 반대편):  D = R + WD
            cos α = (D·cosθ − R) / √(R² + D² − 2RD·cosθ)
        오목(radius < 0, 곡률중심이 카메라 쪽):      d = WD − |R|
            cos α = (d·cosθ + |R|) / √(d² + |R|² + 2d|R|·cosθ)

    두 식은 **중간항 부호만** 다르다. 오목이 완만한 이유가 여기 있다 — 법선이 도는 방향과
    시선이 도는 방향이 상쇄된다. ``|R| = WD`` 면 카메라가 곡률중심에 정확히 놓여 어디를 봐도
    입사각 0°·거리 일정이므로 제한이 없다.

    (오목면의 진짜 한계는 대개 주변 림의 가림과 카메라 몸체 간섭이다 — 그건 가시성 필터와
     cuRobo 충돌 검사가 따로 본다.)
    """
    if not np.isfinite(radius_mm) or radius_mm == 0.0:
        return fov_mm
    if max_incidence_deg <= 0.0 and dof_mm <= 0.0:
        return fov_mm
    convex = radius_mm > 0.0
    R = abs(float(radius_mm))
    D = R + float(wd_mm) if convex else float(wd_mm) - R
    if abs(D) < 1e-9:
        return fov_mm                      # 카메라가 곡률중심 — 제한이 걸리지 않는다
    limits = [fov_mm]

    if max_incidence_deg > 0.0:
        k = math.cos(math.radians(max_incidence_deg)) ** 2
        # a·c² + b·c + e = 0  (c = cosθ). 볼록과 오목은 b 의 부호만 다르다.
        a = D * D
        b = 2.0 * R * D * ((k - 1.0) if convex else (1.0 - k))
        e = R * R * (1.0 - k) - k * D * D
        disc = b * b - 4.0 * a * e
        if disc >= 0.0:
            roots = [(-b - math.sqrt(disc)) / (2.0 * a), (-b + math.sqrt(disc)) / (2.0 * a)]
            thetas = [math.acos(min(1.0, max(-1.0, r))) for r in roots if abs(r) <= 1.0]
            if thetas:
                limits.append(2.0 * R * min(thetas))

    if dof_mm > 0.0:
        # 볼록은 가장자리로 갈수록 **멀어지고**, 오목은 **가까워진다** — 초점을 벗어나는
        # 방향이 반대다.
        target = (float(wd_mm) + dof_mm) ** 2 if convex else (float(wd_mm) - dof_mm) ** 2
        cos_theta = ((R * R + D * D - target) / (2.0 * R * D) if convex
                     else (target - D * D - R * R) / (2.0 * D * R))
        if abs(cos_theta) <= 1.0:
            limits.append(2.0 * R * math.acos(cos_theta))

    return max(1e-3, min(limits))


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


def sample_face(face, o, fov_mm: float, overlap: float,
                min_area_cm2: float = DEFAULT_MIN_FACE_AREA_CM2,
                working_distance_mm: float = 195.0,
                max_incidence_deg: float = 0.0, dof_mm: float = 0.0):
    """면 하나 → (points_mm, normals, info). 격자 + 트리밍 + 해석적 법선."""
    surface = o["BRep_Tool"].Surface_s(face)
    adaptor = o["BRepAdaptor_Surface"](face)
    u0, u1, v0, v1 = o["BRepTools"].UVBounds_s(face)
    props = o["GProp_GProps"]()
    o["BRepGProp"].SurfaceProperties_s(face, props)
    info = {"type": _surface_type_name(face, o), "area_cm2": props.Mass() / 100.0}
    if info["area_cm2"] < min_area_cm2:
        info["skipped"] = "tiny"
        return np.zeros((0, 3)), np.zeros((0, 3)), info

    # 매개변수 → mm 환산은 면 중앙에서 재서 상수로 쓴다. 평면·원통·원뿔은 정확하고,
    # 구·토러스는 근사다(그쪽은 대개 필렛이라 어차피 건너뛴다).
    mid = o["GeomLProp_SLProps"](surface, (u0 + u1) / 2.0, (v0 + v1) / 2.0, 1, 1e-7)
    du_mm = max(mid.D1U().Magnitude(), 1e-9)
    dv_mm = max(mid.D1V().Magnitude(), 1e-9)
    len_u, len_v = (u1 - u0) * du_mm, (v1 - v0) * dv_mm
    # 이 면에서 실제로 쓸 수 있는 프레임 폭. 곡률이 급할수록 좁아진다(평면은 공칭 그대로).
    radius = signed_radius_mm(face, o)
    fov_mm = effective_fov_mm(radius, fov_mm, working_distance_mm,
                              max_incidence_deg, dof_mm)
    info["radius_mm"] = radius
    info["effective_fov_mm"] = fov_mm
    step = fov_mm * (1.0 - overlap)

    def count(length: float, closed: bool) -> int:
        # 닫힌 방향(원통 둘레)은 시작·끝이 이어져 경계가 없다. 열린 방향은 첫 컷이 이미
        # FOV 만큼 덮으므로 나머지를 step 으로 채운다 — '덮을 때까지' 공식.
        if closed:
            return max(1, math.ceil(length / step))
        return max(1, math.ceil((length - fov_mm) / step) + 1)

    n_u = count(len_u, adaptor.IsUClosed())
    n_v = count(len_v, adaptor.IsVClosed())
    us = u0 + (np.arange(n_u) + 0.5) * (u1 - u0) / n_u
    vs = v0 + (np.arange(n_v) + 0.5) * (v1 - v0) / n_v

    classifier = o["BRepTopAdaptor_FClass2d"](face, 1e-6)
    sign = -1.0 if face.Orientation() == o["TopAbs_REVERSED"] else 1.0
    inside_states = (o["TopAbs_IN"], o["TopAbs_ON"])
    points, normals = [], []
    for u in us:
        for v in vs:
            if classifier.Perform(o["gp_Pnt2d"](float(u), float(v))) not in inside_states:
                continue
            local = o["GeomLProp_SLProps"](surface, float(u), float(v), 1, 1e-7)
            if not local.IsNormalDefined():
                continue
            point, normal = local.Value(), local.Normal()
            points.append([point.X(), point.Y(), point.Z()])
            normals.append([sign * normal.X(), sign * normal.Y(), sign * normal.Z()])
    info.update(grid=(n_u, n_v), size_mm=(len_u, len_v), kept=len(points),
                dropped=n_u * n_v - len(points))
    return (np.asarray(points, dtype=np.float64).reshape(-1, 3),
            np.asarray(normals, dtype=np.float64).reshape(-1, 3), info)


def sample_step(step_path, fov_mm: float, overlap: float,
                fillet_max_radius_mm: float = DEFAULT_FILLET_MAX_RADIUS_MM,
                min_area_cm2: float = DEFAULT_MIN_FACE_AREA_CM2,
                part_name: Optional[str] = None, working_distance_mm: float = 195.0,
                max_incidence_deg: float = 0.0, dof_mm: float = 0.0,
                verbose: bool = True):
    """STEP(또는 그 안의 한 부품) → (points_mm, normals, per-face info). 좌표계는 CAD 그대로."""
    faces, o = read_faces(step_path, part_name)
    all_points, all_normals, all_ids, all_fov, infos = [], [], [], [], []
    for index, face in enumerate(faces):
        radius = _fillet_radius(face, o)
        if fillet_max_radius_mm > 0 and radius is not None and radius < fillet_max_radius_mm:
            infos.append({"type": _surface_type_name(face, o), "skipped": "fillet",
                          "radius_mm": radius})
            continue
        points, normals, info = sample_face(
            face, o, fov_mm, overlap, min_area_cm2,
            working_distance_mm=working_distance_mm,
            max_incidence_deg=max_incidence_deg, dof_mm=dof_mm)
        infos.append(info)
        if len(points):
            all_points.append(points)
            all_normals.append(normals)
            # 점마다 '어느 면에서 왔나' 와 '그 면의 유효 FOV' 를 달아 보낸다 — 커버리지
            # 계산과 h5 추적성이 둘 다 이걸 필요로 한다.
            all_ids.append(np.full(len(points), index, dtype=np.int32))
            all_fov.append(np.full(len(points), info.get("effective_fov_mm", fov_mm)))
    points = np.vstack(all_points) if all_points else np.zeros((0, 3))
    normals = np.vstack(all_normals) if all_normals else np.zeros((0, 3))
    face_ids = np.concatenate(all_ids) if all_ids else np.zeros(0, dtype=np.int32)
    point_fov = np.concatenate(all_fov) if all_fov else np.zeros(0)
    if verbose:
        sampled = sum(1 for i in infos if "skipped" not in i)
        fillets = sum(1 for i in infos if i.get("skipped") == "fillet")
        dropped = sum(i.get("dropped", 0) for i in infos)
        print(f"  CAD faces: {len(faces)} (sampled {sampled}, fillet-skipped {fillets}, "
              f"tiny {len(infos) - sampled - fillets})")
        print(f"  Grid points: {len(points)} kept, {dropped} outside trimming boundary")
        narrowed = [i for i in infos
                    if i.get("effective_fov_mm", fov_mm) < fov_mm - 1e-6]
        if narrowed:
            worst = min(i["effective_fov_mm"] for i in narrowed)
            print(f"  Curvature-limited FOV on {len(narrowed)} face(s): "
                  f"down to {worst:.1f}mm (nominal {fov_mm:.0f}mm)")
    return points, normals, {"infos": infos, "face_id": face_ids,
                             "effective_fov_mm": point_fov}


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


def sample_step_aligned(step_path, reference_mesh, fov_mm: float, overlap: float,
                        fillet_max_radius_mm: float = DEFAULT_FILLET_MAX_RADIUS_MM,
                        tol_linear: Optional[float] = None, align: str = ALIGN_AUTO,
                        part_name: Optional[str] = None, working_distance_mm: float = 195.0,
                        max_incidence_deg: float = 0.0, dof_mm: float = 0.0,
                        verbose: bool = True):
    """``sample_step`` 결과를 파이프라인 좌표계(미터)로 옮겨 돌려준다."""
    points_mm, normals, meta = sample_step(
        step_path, fov_mm, overlap, fillet_max_radius_mm,
        part_name=part_name, working_distance_mm=working_distance_mm,
        max_incidence_deg=max_incidence_deg, dof_mm=dof_mm, verbose=verbose)
    transform, score = cad_to_reference_transform(
        step_path, reference_mesh, tol_linear, align=align, part_name=part_name)
    rotation = transform[:3, :3]
    points = (rotation @ (points_mm.T / 1000.0)).T + transform[:3, 3]
    normals = (rotation @ normals.T).T
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths < 1e-9, 1.0, lengths)
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

    points, normals, areas, face_of = [], [], [], []
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
                jac = np.linalg.norm(np.cross(
                    [local.D1U().X(), local.D1U().Y(), local.D1U().Z()],
                    [local.D1V().X(), local.D1V().Y(), local.D1V().Z()]))
                areas.append(jac * ((u1 - u0) / n_u) * ((v1 - v0) / n_v))
                points.append([point.X(), point.Y(), point.Z()])
                normals.append([sign * normal.X(), sign * normal.Y(), sign * normal.Z()])
                face_of.append(index)
    if not points:
        empty = np.zeros((0, 3))
        return SurfaceCells(empty, empty, np.zeros(0), np.zeros(0, dtype=np.int32), transform)

    rot, offset = transform[:3, :3], transform[:3, 3]
    P = (rot @ (np.asarray(points).T / 1000.0)).T + offset       # CAD(mm) → 파이프라인(m)
    N = (rot @ np.asarray(normals).T).T
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return SurfaceCells(points=P, normals=N,
                        areas_cm2=np.asarray(areas) / 100.0,     # mm² → cm²
                        face_id=np.asarray(face_of, dtype=np.int32),
                        transform=transform)


def inspectable(cells: SurfaceCells, bottom_angle_deg: float = 0.0,
                rotation=None) -> np.ndarray:
    """검사 **대상**인 셀 = True. 아래를 향해 로봇이 볼 수 없는 셀은 분모에서 뺀다.

    알고리즘의 실패("구멍")와 물리적 불가("애초에 못 보는 곳")를 섞지 않기 위한 구분이다.
    판정 규칙은 bottom filter 와 같은 것을 쓴다(visibility.facing_up).
    """
    return visibility.facing_up(cells.normals, rotation, bottom_angle_deg)


def coverage_report(step_path, positions, normals, point_fov_mm, *,
                    working_distance_mm: float, occluder_mesh=None,
                    max_incidence_deg: float = 0.0, dof_mm: float = 0.0,
                    bottom_angle_deg: float = 0.0, rotation=None,
                    part_name: Optional[str] = None, tol_linear: Optional[float] = None,
                    align: str = ALIGN_AUTO, reference_mesh=None,
                    cell_mm: float = DEFAULT_COVERAGE_CELL_MM,
                    cells: Optional[SurfaceCells] = None,
                    verbose: bool = True) -> dict:
    """생성된 viewpoint 가 각 CAD 면을 얼마나 덮는지 면적 가중으로 센다.

    셀은 셋 중 하나다:
      * **covered**  — 어떤 viewpoint 가 검사 가능한 조건으로 본다
      * **검사 대상 아님** — 아래를 향해 로봇이 볼 수 없음. 분모에서 뺀다
      * **uncovered** — 덮을 수 있는데 안 덮인 곳. **이것만이 진짜 구멍이다**

    판정은 ``visibility`` 가 전담한다 — 필터·선택과 같은 코드를 쓰므로 셋이 어긋날 수 없다.
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
    target = inspectable(cells, bottom_angle_deg, rotation)
    covered = visibility.covered_by_any(
        cells.points, cells.normals,
        np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        np.asarray(normals, dtype=np.float64).reshape(-1, 3),
        point_fov_mm, occluder_mesh, spec, mask=target)

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
              "cells": int(len(cells))}
    if verbose:
        print(f"  Coverage: {result['covered_ratio']*100:.1f}% "
              f"({total_c:.1f}/{total_t:.1f} cm² of inspectable area, "
              f"{result['cells']} cells)")
        for index, row in sorted(report.items(), key=lambda kv: -kv[1]["target_cm2"])[:6]:
            if row["target_cm2"] < 0.5:
                continue
            print(f"    face {index:3d}: {row['ratio']*100:5.1f}% "
                  f"({row['covered_cm2']:6.1f}/{row['target_cm2']:6.1f} cm²)")
    return result

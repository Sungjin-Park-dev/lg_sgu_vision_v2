"""Data models shared by viewpoint generation and downstream consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from common import config


DEFAULT_DELAUNAY_NEIGHBORS = 12
DEFAULT_DELAUNAY_DISTANCE_FACTOR = 2.5
# 90° = "두 법선이 서로 반대 반구를 향하면 자른다" — 물체를 관통하는 간선을 막는
# 물리적 경계다. 예전 75° 는 그보다 15° 더 조여서, 모서리처럼 꺾인 실제 표면까지
# 잘라 viewpoint 그래프를 조각냈다. 실측(sample/74): 75° 는 윗면 57개와 측면 17개가
# 따로 놀아 두 조각 사이를 잇는 transit 하나가 궤적 전체 base 회전의 69% 를 차지했다.
# 90° 로 올리면 간선 2개가 더해지며 한 덩어리가 되고 base 회전이 613° -> 285° 다.
# 다른 물체 확인: curved 무변화, cylinder 성분 그대로, square 3 -> 2조각.
DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG = 90.0


@dataclass(frozen=True)
class ViewpointAdjacency:
    """Optional local-surface adjacency stored below ``viewpoints/adjacency``.

    ``edges``가 그래프의 유일한 진실이다. 연결성분이 필요하면
    ``adjacency.components_from_edges(edges, n)``로 파생한다 — 저장하지 않는다.
    """

    edges: np.ndarray
    method: str
    stats: dict[str, object]
    k_neighbors: int | None = None
    distance_factor: float | None = None
    max_normal_angle_deg: float | None = None


@dataclass(frozen=True)
class ViewpointData:
    """Canonical in-memory representation of a viewpoint HDF5 file.

    세 계층으로 고정된다 — 기하(positions/normals/working_distance), 표면 그래프
    (adjacency), 선택된 그룹핑과 방문 순서(cluster_id/cluster_order/path_order).
    그룹핑을 만든 방법은 ``metadata/clustering_method``에만 기록되고, 어떤 방법이든
    이 세 계층의 모양은 동일하다.
    """

    source_path: Path
    positions: np.ndarray
    normals: np.ndarray
    path_order: np.ndarray | None
    cluster_id: np.ndarray | None
    cluster_order: np.ndarray | None
    adjacency: ViewpointAdjacency | None
    input_mesh: str | None
    working_distance_m: float
    # Camera-spec snapshot captured at generation time (metadata/camera_spec),
    # so preview/execute can default the FOV visualization to what the
    # viewpoints were actually planned with. Fall back to config when absent.
    fov_width_m: float
    fov_height_m: float

    @property
    def count(self) -> int:
        return int(len(self.positions))

    # 저장은 m, 카메라 스펙의 표기 단위는 mm(h5 metadata/camera_spec, GUI 입력칸, config
    # 상수가 전부 mm)다. 그 경계를 여기 세 property 로 모아 둔다 — 없으면 읽는 쪽마다
    # ``* 1000.0`` 이 붙는다.
    @property
    def working_distance_mm(self) -> float:
        return self.working_distance_m * 1000.0

    @property
    def fov_width_mm(self) -> float:
        return self.fov_width_m * 1000.0

    @property
    def fov_height_mm(self) -> float:
        return self.fov_height_m * 1000.0

    @property
    def camera_spec(self) -> dict:
        """h5 ``metadata/camera_spec`` 과 같은 모양 — ``ViewpointGenParams.camera_spec`` 의 읽기 짝."""
        return {
            "fov_width_mm": self.fov_width_mm,
            "fov_height_mm": self.fov_height_mm,
            "working_distance_mm": self.working_distance_mm,
        }

    @property
    def visit_indices(self) -> np.ndarray:
        """Indices in stored visit order; legacy files fall back to HDF5 order."""
        if self.path_order is None:
            return np.arange(self.count, dtype=np.int32)
        return np.argsort(self.path_order, kind="stable").astype(np.int32)


@dataclass
class ViewpointGenParams:
    """Parameters for the importable viewpoint generation pipeline.

    카메라 스펙(FOV·WD·overlap)은 ``None`` 이면 ``__post_init__`` 에서 config 기본값으로
    해소된다 — config 는 기본값이고, **이 객체가 그 실행의 진실**이다. 덕분에 하류는
    ``params.working_distance_mm`` 를 무조건 float 로 읽을 수 있고, 생성기는 config 전역을
    다시 들여다볼 필요가 없다.
    """

    material_rgb: Optional[str] = None
    color_tolerance: float = 5.0
    row_spacing_mm: Optional[float] = None
    col_spacing_mm: Optional[float] = None
    # 카메라 스펙 — 저장 시 metadata/camera_spec 으로 h5 에 박히고, 그 h5 를 읽는
    # IK/궤적/GLNS/Isaac 이 config 대신 그 값을 쓴다(docs/reference/camera-geometry.md).
    working_distance_mm: Optional[float] = None
    fov_width_mm: Optional[float] = None
    fov_height_mm: Optional[float] = None
    overlap_ratio: Optional[float] = None
    filter_bottom: bool = True
    bottom_angle: float = 80.0
    filter_interior: bool = False
    interior_hull_align_min: float = 0.3
    # 샘플링은 메시 표면 직접 FPS 하나뿐이다(grid 모드는 2026-08-26 제거).
    surface_spacing_mm: Optional[float] = None
    build_delaunay: bool = True
    delaunay_neighbors: int = DEFAULT_DELAUNAY_NEIGHBORS
    delaunay_distance_factor: float = DEFAULT_DELAUNAY_DISTANCE_FACTOR
    delaunay_max_normal_angle_deg: float = DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG

    def __post_init__(self):
        """카메라 스펙 미지정분을 config 기본값으로 해소한다(생성 시점 1회).

        항상 float 로 통일한다 — GUI/CLI 에서 int 가 들어오면 h5 attr 이 int64 로 박혀
        같은 스펙인데 파일마다 dtype 이 갈린다.
        """
        self.working_distance_mm = float(
            config.CAMERA_WORKING_DISTANCE_MM if self.working_distance_mm is None
            else self.working_distance_mm)
        self.fov_width_mm = float(
            config.CAMERA_FOV_WIDTH_MM if self.fov_width_mm is None else self.fov_width_mm)
        self.fov_height_mm = float(
            config.CAMERA_FOV_HEIGHT_MM if self.fov_height_mm is None else self.fov_height_mm)
        self.overlap_ratio = float(
            config.CAMERA_OVERLAP_RATIO if self.overlap_ratio is None else self.overlap_ratio)

    @property
    def camera_spec(self) -> dict:
        """h5 ``metadata/camera_spec`` 에 그대로 들어가는 dict (storage 의 읽기 키와 일치)."""
        return {
            "fov_width_mm": self.fov_width_mm,
            "fov_height_mm": self.fov_height_mm,
            "working_distance_mm": self.working_distance_mm,
        }


@dataclass
class ViewpointResult:
    """In-memory generation result; persistence remains the caller's choice.

    방문 순서와 클러스터 라벨은 없다 — 순서는 GLNS 가 IK 자세와 함께 푼다.
    ``nn_path_length_mm`` 은 greedy nearest-neighbor 베이스라인(보고용)일 뿐
    실행 순서가 아니다.
    """

    positions: np.ndarray
    normals: np.ndarray
    camera_positions: np.ndarray
    row_spacing_m: float
    col_spacing_m: float
    nn_path_length_mm: float
    adjacency: Optional[dict]

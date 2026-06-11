# square_structure 안쪽 벽 viewpoint 문제 (2026-06-11, 보류)

## 문제 요약

viser 스튜디오에서 `square_structure`를 생성하면 viewpoint가 **겉/윗면뿐 아니라 안쪽 면에도**
생긴다. 사용자 요구는 "mesh 겉/윗면에만" 생성. 처음 가설은 "메시가 제대로 안 닫혀서"였으나,
조사 결과 **닫힘이 아니라 얇은 벽 속 빈 박스라 안쪽 벽까지 표면 샘플링되는 것**이 원인.

## 진단

### 메시는 (파이프라인에선) 닫혀 있다

| 로드 방식 | components | watertight | 비고 |
|-----------|-----------|------------|------|
| `trimesh.load(process=False)` | 9379 | False | 중복 정점 미병합 = triangle soup |
| `trimesh.load(process=True)` (= `load_meshes` 기본) | 1 | **True** | 정점 자동 병합, winding 일관, 경계 엣지 0 |

즉 **파이프라인이 받는 메시는 watertight**. `process=False`로 보면 안 닫힌 것처럼 보이는 건
중복 정점 때문이고, 닫힘 자체는 원인이 아니다.

### 진짜 원인 = 얇은 벽 속 빈 박스 (안쪽 벽 63%)

`load_meshes('square_structure')`(watertight) 기준:

| 지표 | 값 | 의미 |
|------|-----|------|
| `mesh.area / convex_hull.area` | **1.78** | 겉 envelope보다 표면이 1.78배 = 안쪽 벽 존재 |
| `mesh.volume / hull.volume` | **4.1%** | 속이 거의 빈 얇은 벽 박스 (벽 두께 ~1.2mm) |
| 중심 향해 안쪽 보는 면 비율 | **63%** | 면의 절반 이상이 안쪽 벽 |
| 면 중심 z 히스토그램 | `[6622,0,140,140,0,6622]` | 위/아래면 대부분, 측벽 ~280 |
| extents | `[0.26, 0.102, 0.03] m` | 260×102×30mm 박스 |

`sample_surface_even`이 겉면 + **안쪽 벽(63%)**을 모두 샘플링 → viewpoint가 안쪽에도 생김.

> ⚠️ `mesh.contains(camera_positions)`로는 **4%만** 잡힌다. 공동(cavity)은 "솔리드 내부"가
> 아니라 빈 void라서, 안쪽 벽을 향하는 카메라도 contains=False. **contains 기반 필터는 무효.**

## 시도한 해법

### ① voxel 겉껍질 추출 — 실패

`mesh.voxelized(pitch).fill().marching_cubes`로 외피만 추출 시도.

| pitch | 결과 |
|-------|------|
| 2mm | marching_cubes가 **voxel 인덱스 좌표** 반환(단위 깨짐, `apply_transform(vox.transform)` 필요) + `fill()`이 새서 `area/hull` 1.78→1.74로 거의 안 줄어듦 + 11.8s |
| 1mm / 0.7mm | **OOM (exit 137)** |

원인: **벽 두께(~1.2mm) < voxel pitch** → 벽이 voxel 장벽을 못 만들어 `fill()`이 공동을 못 채우고
샘. pitch를 벽보다 얇게 하면 메모리 폭발. → **얇은 벽 객체엔 voxel solidify 부적합.**

### ② 외부 가시성/도달성 ray 필터 — 부분적 (가장 유망하나 미완)

watertight 메시에서 표면 4000점 샘플(내부 라벨 = 법선이 중심 향함, dot>0.25):
inner-wall 583 / outer 1788.

| 필터 | 유지 | 안쪽 벽 오염 | 바깥면 손실 |
|------|------|-------------|------------|
| A. 표면점에서 +법선 ray 탈출 | 1358 | 9% | **31%** |
| B. 카메라에서 +법선 ray 탈출 | 2266 | 21% | 0% |
| C. 카메라→점 LOS 첫 hit≈wd | 1463 | 15% | 31% |
| D. A∧C | 1358 | 9% | 31% |

깔끔하게 안 갈림: 오염을 줄이면(A/C) 바깥면 31% 손실, 손실을 없애면(B) 오염 21%.
바깥면 손실 31%의 원인 추정 = **겉면 자체에 홈/리세스**가 있어 +법선 ray가 맞은편 겉면에 막힘.

## 결론 / 보류

- voxel은 폐기(얇은 벽). 외부 가시성 필터가 물리적으로 옳은 방향이나, 단순 단일 ray로는
  오염/손실 트레이드오프가 커서 **추가 설계 필요**.
- 재개 시 후보:
  1. **다중 ray openness**(카메라에서 구면 N방향, 탈출 비율 임계) + 임계 튜닝
  2. **렌더 기반 면 가시성**: 외부 다수 시점에서 ortho 렌더 → 한 번이라도 보이는 면 = 겉면
     (표준 exterior surface extraction, 견고하나 무거움)
  3. **업스트림 메시 정리**: 애초에 안쪽 벽 없는 깨끗한 외피 솔리드를 소스로 받기
     (voxel 외 solidify, 또는 CAD 재익스포트)
- 영향 범위: 이 문제는 **얇은 벽 속 빈/리세스 객체**에 한함. `sample`(열린 단면 스캔),
  곡면/원통 등 단일 외피 객체엔 무관.

## 재현

```bash
uv run scripts/viser/viewpoint_app.py --object square_structure
# Sampling=surface, Sub-cluster=agglomerative 로 Generate → 안쪽 면에 viewpoint 확인
```

진단 스니펫(요지):
```python
from pipeline.generate_viewpoints import load_meshes
full, m, _ = load_meshes('square_structure')      # watertight=True
m.area / m.convex_hull.area                         # 1.78  (안쪽 벽 존재)
m.volume / m.convex_hull.volume                     # 0.041 (얇은 벽 속 빈 박스)
```

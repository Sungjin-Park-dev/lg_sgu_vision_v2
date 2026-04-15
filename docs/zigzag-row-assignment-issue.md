# 지그재그 행 할당 문제 (2026-04-08)

## 문제 요약

`reorder_zigzag()`에서 클러스터 내부 포인트들의 **행(row) 할당**이 곡면에서 잘못되어,
지그재그 경로가 되돌아가는 현상이 발생한다.

## 현재 코드 (양자화)

```python
# scripts/generate_viewpoints.py, reorder_zigzag() 내부
row_index = np.round((proj1 - proj1.min()) / max(row_spacing_m, 1e-9)).astype(np.int32)
```

PCA axis1에 투영한 proj1 값을 `row_spacing_m`(15mm) 간격으로 양자화하여 행을 배정한다.

## 문제 재현

```bash
uv run scripts/generate_viewpoints.py --object sample --material-rgb "170,163,158" \
  --cluster-method coacd+dbscan --normal-weight 0.05 --coacd-threshold 0.25 --compare
```

`coacd+dbscan t=0.25, eps=41mm` 설정에서 cluster 15 (16개 포인트)를 확인하면:

```
포인트   proj1     양자화값   round   할당 행
VP38     +0.0072    1.40       1       row 1
VP43     +0.0079    1.45       1       row 1
VP58     +0.0085    1.49       1       row 1   ← 1.5 미만
VP63     +0.0092    1.53       2       row 2   ← 1.5 이상
VP78     +0.0098    1.57       2       row 2
...
```

7개의 연속된 포인트(proj1: 0.0072~0.0111, 내부 gap ~0.7mm)가 `round()` 경계 1.5에 걸려
**row 1(3개)과 row 2(4개)로 쪼개진다.**

결과적으로 지그재그가:
```
row 1(홀수←): VP58 ← VP43 ← VP38
                │
row 2(짝수→): VP98 → VP83 → VP78 → VP63   ← 되돌아감!
```
step 10→11에서 VP58(proj2=+0.009) → VP98(proj2=-0.051)으로 **60mm 점프하며 되돌아간다.**

## 시도한 해결 방법

### 1. Gap 기반 (고정 threshold = row_spacing × 0.5)

```python
sorted_idx = np.argsort(proj1)
gaps = np.diff(proj1[sorted_idx])
# gap > 7.5mm 인 곳에서만 행 분리
```

- **작은 클러스터 (16pts)**: 행 내부 gap ~0.7mm, 행 간 gap ~16mm → 정상 분리 (3행)
- **큰 클러스터 (49pts)**: 행 간 gap ~5.8mm < threshold 7.5mm → 전부 1행으로 합침 (깨짐)
- **그리드 (124pts)**: 행 간 gap ~5.75mm < threshold 7.5mm → 1행으로 합침 (깨짐)

### 2. Median gap 기반 (threshold = median × 3)

```python
median_gap = np.median(gaps)
threshold = median_gap * 3
```

- **작은 클러스터**: median=0.65mm, threshold=1.95mm → 정상 (3행)
- **큰 클러스터 (49pts)**: median=2.16mm, threshold=6.48mm, max_gap=5.84mm → 1행 (깨짐)
- **그리드**: median=1.22mm, threshold=3.67mm → 9행 (원래 12-13행이어야 함)

## 근본 원인

곡면 위의 클러스터에서 PCA 축에 투영하면, 행 내부 gap과 행 간 gap의 차이가
클러스터 크기에 따라 달라진다:

| 상황 | 행 내부 gap | 행 간 gap | 비율 |
|------|-----------|----------|------|
| 작은 클러스터 (16pts, 좁은 곡면) | ~0.7mm | ~16mm | 23× |
| 큰 클러스터 (49pts, 넓은 곡면) | ~2mm | ~5mm | 2.5× |
| 그리드 전체 (124pts) | ~1mm | ~4-6mm | 3-5× |

작은 클러스터에서는 bimodal 분포 (행 내부 vs 행 간이 뚜렷하게 구분)지만,
**큰 곡면 클러스터에서는 gap이 연속적으로 분포**하여 어떤 고정/적응형 threshold로도
안정적인 경계를 찾기 어렵다.

## 해결 (2026-04-09): 원본 grid row_index 전달

### 원인 분석

`generate_grid_viewpoints()`에서 그리드 생성 시 정확한 `row_index`가 이미 확정되는데,
`compute_cluster_internal_order()` → `reorder_zigzag()`에서 클러스터별 PCA 투영 + 양자화로
행을 **다시 추정**하면서 정보가 소실되고 있었다.

```
generate_grid_viewpoints()          compute_cluster_internal_order()
 row_index = i (정확)  ──(덮어씀)──→ row_index = round(proj1/spacing) (근사, 오류 발생)
```

### 변경 내용

1. **`reorder_zigzag()`**: `row_index_override` 파라미터 추가.
   제공 시 양자화 대신 외부 행 인덱스를 그대로 사용.

2. **`compute_cluster_internal_order()`**: `grid_row_index`, `global_axis2` 파라미터 추가.
   클러스터별 포인트의 원본 행 인덱스를 `reorder_zigzag()`에 전달.
   행 내 정렬은 전역 axis2(열 방향)를 사용.

3. **main flow**: `grid_row_index = row_index.copy()`로 원본 보존.
   필터링(bottom-facing 제거) 시 함께 필터링.
   `run_clustering()` 내에서 `compute_cluster_internal_order()`에 전달.

### 검증 결과 (cluster 15, 16pts)

```
[OLD] 4 rows, max step: 59.8mm, total: 313.6mm
[NEW] 8 rows, max step: 43.2mm, total: 295.8mm

step       OLD       NEW
 0->1     14.9     43.2
 ...
10->11    59.8     20.4   ← 60mm 점프 해소
 ...
14->15    49.4     20.4   ← 49mm 점프 해소
```

- 원본 그리드의 8개 행이 정확히 복원됨
- 양자화 경계(1.5)에서의 오분류 문제가 원천적으로 해결됨
- step 0→1이 43.2mm인 것은 row 2에 포인트가 1개만 있어서 발생하는 행 전환 거리

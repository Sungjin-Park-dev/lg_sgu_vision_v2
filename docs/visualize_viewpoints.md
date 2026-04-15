# visualize_viewpoints.py

클러스터링된 뷰포인트 경로를 Plotly 기반 인터랙티브 3D HTML로 출력한다.  
`generate_viewpoints.py`에서 import하여 사용한다.

## 함수

| 함수 | 설명 |
|------|------|
| `visualize_clusters_html` | 메시 + 클러스터별 뷰포인트/경로를 HTML로 저장. 드롭다운으로 여러 모드 비교 지원 |
| `_build_cluster_mode_traces` | 단일 클러스터링 모드에 대한 plotly traces 생성 (내부 함수) |

## 시각화 요소

- **메시**: 반투명 회색 배경
- **클러스터별 뷰포인트**: 고유 색상 마커 + 클러스터 내 경로 라인
- **클러스터 간 이동**: 회색 점선
- **CoACD 파트 메시**: 반투명 색상 오버레이 (coacd, coacd+dbscan 모드)

## 드롭다운 비교 모드

`--compare` 플래그 사용 시, 여러 파라미터 조합 결과를 드롭다운으로 전환하며 비교할 수 있다.  
버튼 라벨에 경로 길이, 클러스터 수, 최적 대비 차이(%)가 표시된다.

## 의존성

- `plotly` — 3D 인터랙티브 시각화
- `numpy` — 배열 연산

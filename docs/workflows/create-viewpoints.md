# 뷰포인트 만들기

Viewpoint Studio에서 물체 표면의 검사 위치와 카메라 방향을 생성하고 HDF5로 저장한다.

## 실행

```bash
uv run --no-sync scripts/apps/viewpoint_studio.py --object sample
```

브라우저에서 `http://localhost:8080`에 접속한다. 기존 파일을 바로 열려면 `--viewpoints PATH`를 사용한다.

## 작업 순서

1. `Object`에서 물체를 고른다.
2. `FOV`의 overlap과 클러스터 설정을 조절한다.
3. `Generate`로 뷰포인트를 만든다.
4. 레이어와 `Playback`으로 분포와 순서를 확인한다.
5. `Save h5`로 저장한다.

## 주요 기능

| 영역 | 기능 |
|---|---|
| `Existing h5` | 저장된 뷰포인트 다시 열기 |
| `Layers` | 메시, 경로, 전환, Delaunay 인접 그래프 표시 |
| `FOV` | 카메라 중첩률로 표면 간격 조절 |
| `CoACD + sub-cluster` | 물체 형상별 영역 분할과 내부 클러스터링 |
| `Playback` | 최종 검사 순서 재생 |

출력은 `data/{object}/viewpoint/{N}/viewpoints_{method}.h5`에 저장된다. 새 물체가 목록에 없다면 [자산 준비](../guides/prepare-object-assets.md)를 먼저 확인한다.

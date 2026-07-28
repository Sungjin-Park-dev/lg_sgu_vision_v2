# 물체 배치 스윕 결과

`scripts/common/config.py` 의 `OBJECT_PLACEMENTS` 가 왜 그 좌표인지의 **근거 자료**다.

물체를 로봇 앞 격자 위치에 옮겨가며 GLNS 를 돌려, 도달 가능한 viewpoint 수와 base reconfig
횟수를 측정한 결과다. 채택된 배치는 전부 base reconfig = 0 인 지점이다.

물체별로:

| 파일 | 내용 |
|---|---|
| `summary.csv` | 격자 위치별 도달률·reconfig·성분 수 |
| `summary.json` | 같은 내용 + 스윕 설정 |
| `heatmap_z{0,1,2}.png` | 높이별 도달률 히트맵 |

스윕 raw 산출물(위치별 해 h5 와 solve 로그)은 재현 가능하고 용량만 차지해 보관하지 않는다.
스윕을 돌린 도구(`scripts/tools/optimize_placement.py`)는 현재 저장소에 없다 — research
브랜치 전용이었다. 다시 필요하면 그쪽 이력에서 가져온다.

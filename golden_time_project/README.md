# 골든타임 분석 파이프라인

콘텐츠(영화) 흥행 → 실제 방문 급증까지의 "골든타임"을 계산·검증하는 코드. 전체 프로젝트 배경과 최종 결과는 [상위 폴더 README](../README.md)와 [`../골든타임_프로젝트_최종정리.md`](../골든타임_프로젝트_최종정리.md) 참고.

## 설치

```bash
cd golden_time_project
pip install -r requirements.txt
```

## 실행 (아래 순서대로 실행하면 최종 결과가 재현됨)

```bash
python golden_time_calculator.py      # ① 사례별 골든타임 갭(검색 급증일 → 공식대응/방문피크일) 계산
python did_analysis.py                # ② 대조군 비교(DID)로 순수 콘텐츠 효과 검증
python anomaly_classifier_final.py    # ③ 최종 채택 분류 모델 성능(LOCO AUC) 산출
python transfer_learning_chronos.py   # ④ 사전학습 시계열 모델(Chronos)로 ①의 결과를 독립적으로 교차검증
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `config.py` | 사례별 메타데이터(개봉일·배경지·사이트 키워드 등) |
| `convert_teammate_daily.py` | 팀원 제공 일별 원본(`영화관광_학습용_추가_완성본.xlsx`)을 이 파이프라인이 읽는 CSV 형식으로 변환 |
| `golden_time_calculator.py` | **규칙기반 골든타임 계산기** — 평시 대비 표준편차 N배 초과일 탐지 + 민감도 분석. 실전 사용 중인 결과물 |
| `did_analysis.py` | **이중차분법(DID)** — 대조군과 비교해 계절/전국 추세를 제거한 순수 콘텐츠 효과 산출 |
| `transfer_learning_chronos.py` | 사전학습 시계열 모델(Amazon Chronos)로 재학습 없이 이상 탐지, 규칙기반 결과 교차검증 |
| `anomaly_classifier_final.py` | **최종 채택 분류 모델** — "이 지역-이 시기가 콘텐츠 효과로 튄 것인지" 판정 (LOCO AUC 0.786) |
| `experiments/` | 분류 모델 성능 개선 과정에서 시도한 이전 버전들(v1~v4) — 실패한 시도까지 정직하게 남겨둔 기록. 최종 결론은 `anomaly_classifier_final.py`에 정리된 것이 맞음 |
| `data/raw/` | 정제된 CSV 원자료 (Google Trends, 관광데이터랩 방문자수, KOBIS 박스오피스 등) |

## 핵심 결과 요약

- 영화 흥행 시 배경지 방문객은 대조군 대비 평균 **+12.1%p** 증가 (랜덤효과 메타분석, 95% CI +2.4~+21.8%p)
- "이 지역-이 시기가 콘텐츠 효과로 튄 것인지" 판정 모델: **LOCO AUC 0.786** (0.5=무작위, 1.0=완벽)

자세한 사례별 수치와 신뢰도 평가는 [`../골든타임_프로젝트_최종정리.md`](../골든타임_프로젝트_최종정리.md) 참고.

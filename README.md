# 콘텐츠 골든타임 프로젝트

2026 한국관광 데이터랩 활용 경진대회 출품작 — **콘텐츠(영화) 흥행 → 실제 방문 급증까지의 "골든타임" 예측 모델**

---

## 프로젝트 핵심 결과

**한 줄 요약**: 영화 흥행 시, 배경지 방문객은 평균 약 +12%p 증가(대조군 비교로 검증, 95% CI +2.4~+21.8%p). 영월·자산어보 2개 사례는 강하게 검증됨.

---

## 폴더 구조

```
travel/
├── README.md                              ← 이 파일
├── 골든타임_프로젝트_최종정리.md            ← 제출용 핵심 결과 (기술 버전)
├── 골든타임_프로젝트_쉬운설명.md            ← 제출용 핵심 결과 (쉬운 버전)
├── 코딩_작업_현황.md                        ← 작업 로그
├── 초기_아이디에이션_초안.md / 아이디어_구체화_실행계획.md / 발표_대본_초안.md / 0831회의.../0921...  ← 과거 회의·기획 기록
├── 영화관광_*.xlsx                          ← 팀원 제공 원본 데이터/수집표
├── (첨부1)(공모요강)*.pdf / (첨부2)(참가서류)*.zip ← 대회 공식 서류 양식
└── golden_time_project/
    ├── golden_time_calculator.py           ← 규칙기반 골든타임 계산기 (실전 사용)
    ├── did_analysis.py                     ← 대조군 비교(DID) 분석
    ├── anomaly_classifier_final.py         ← 최종 채택 분류 모델 (control_z, AUC 0.786)
    ├── anomaly_classifier.py / _v2 / _v3 / _v4  ← 분류 모델 개선 과정 전체 기록
    ├── transfer_learning_chronos.py        ← 전이학습(Chronos) 실험
    ├── convert_teammate_daily.py           ← 팀원 제공 일별 데이터 변환기
    ├── config.py                           ← 사례별 메타데이터(개봉일·촬영지 등)
    └── data/raw/                           ← 정제된 CSV 데이터 (Google Trends, 방문자수, 박스오피스 등)
```

---

## 실행 방법

```bash
cd golden_time_project
python golden_time_calculator.py      # 사례별 골든타임 갭 계산
python did_analysis.py                # 대조군 비교(DID) 결과
python anomaly_classifier_final.py    # 최종 분류 모델 성능(AUC)
```

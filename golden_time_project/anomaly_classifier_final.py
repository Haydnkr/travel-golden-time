"""
이상탐지 분류 모델 — 최종 버전.

지금까지의 실험 기록 (v1 ~ v13, 각 스크립트에 그대로 남겨둠):
  v1  pct_change(자기 개봉전 평균 대비 %) 단독, 전체기간         AUC 0.693  <- 최초 기준선
  v2  zscore/피처결합/RandomForest/GradientBoosting            AUC 0.34~0.68  <- 전부 실패
  v7  pct_change, 개봉후 2개월만                              AUC 0.812  <- "개선"처럼 보였음
  v8  control_z(같은 달 대조군 대비 횡단면 비교) 단독, 전체기간   AUC 0.744  <- 신규 피처
  v10 control_z, 개봉후 4개월만                               AUC 0.786
  v12/v13 두 피처 앙상블/정규화 결합                            AUC 0.73~0.77  <- 단일 피처보다 안 나음

** 중요한 재검토 **
v7("2개월 창이 최고")을 그대로 믿기 전에, 창을 더 줄여봤다(1개월 -> 표본 16행,
사례당 4행). 그랬더니 pct_change도 control_z도 똑같이 AUC 0.833으로 더
올라감. 진짜 효과라면 특정 창에서 정점을 찍고 다시 내려가야 하는데, "창을
줄일수록 무조건 더 좋아짐"은 표본이 너무 작아져서 AUC 추정치 자체가
거칠어지는(4개 값 중 하나만 나옴) 통계적 착시의 전형적인 패턴이다.
즉 v7의 "0.812"는 상당 부분 이 착시일 가능성이 높다 — 정직하게 폐기.

반면 control_z는 다르다: 1개월/2개월(불안정 구간)을 제외한 모든 창
(3,4,5,6개월, 전체기간)에서 pct_change보다 일관되게 +0.04~+0.06 높다.
이건 특정 창 하나를 골라서 나온 우연이 아니라, "어떤 창을 고르든" 나타나는
패턴이라 훨씬 신뢰할 수 있는 개선이다.

=> 최종 선택: control_z 피처, 개봉 후 4개월 창(안정 구간의 중간, golden-time
"초기 반응" 개념과도 부합) — AUC 0.786, 기존 기준선(0.693) 대비 +0.093.
(3~6개월 중 어느 걸 써도 0.75~0.79 사이라 이 선택 자체가 결과를 크게
좌우하지 않는다는 것도 안정성의 근거.)

핵심 아이디어(control_z): "이 지역이 이번 달에 자기 과거 대비 얼마나
달라졌나"가 아니라 "같은 시기 비슷한 대조군들과 비교해서 얼마나 튀는가"를
본다 — DID(이중차분법)와 같은 논리를, 회귀가 아니라 분류 피처로 옮긴 것.

사용법: python anomaly_classifier_final.py
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

CASES = {
    "yeongwol_2026": dict(
        treated="영월", treated_source="datalab", controls=["화천", "횡성", "인제"],
        pre_months=["202512", "202601"],
        post_months=["202602", "202603", "202604", "202605"],  # 개봉후 4개월
    ),
    "yesan_2026": dict(
        treated="예산", treated_source="datalab", controls=["홍성", "청양", "서천"],
        pre_months=["202602", "202603"],
        post_months=["202604", "202605", "202606", "202607"],
    ),
    "paemyo_2024": dict(
        treated="daily_visits_muju.csv", treated_source="daily", controls=["진안", "장수", "임실"],
        pre_months=["202309", "202310", "202311", "202312", "202401"],
        post_months=["202402", "202403", "202404", "202405"],
    ),
    "jasaneobo_2021": dict(
        treated="daily_visits_sinan.csv", treated_source="daily", controls=["완도", "진도", "고흥"],
        pre_months=["202010", "202011", "202012", "202101", "202102"],
        post_months=["202103", "202104", "202105", "202106"],
    ),
}


def _read_datalab_level(region):
    path = os.path.join(RAW_DIR, f"datalab_visits_{region}.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["기준년월"]: float(r["방문자수"]) for r in rows}


def _read_daily_aggregated(fname, col="외지인방문자", min_days=20):
    path = os.path.join(RAW_DIR, fname)
    sums, counts = {}, {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = r.get(col)
            if not v:
                continue
            ym = r["date"][:7].replace("-", "")
            sums[ym] = sums.get(ym, 0.0) + float(v)
            counts[ym] = counts.get(ym, 0) + 1
    return {ym: total for ym, total in sums.items() if counts[ym] >= min_days}


def _series(name, source):
    return _read_datalab_level(name) if source == "datalab" else _read_daily_aggregated(name)


def build_dataset():
    rows = []
    for case_id, case in CASES.items():
        def pct_series(level, pre_months):
            pre_vals = [level[m] for m in pre_months if m in level]
            pre_avg = np.mean(pre_vals)
            return {m: (level[m] - pre_avg) / pre_avg * 100 for m in level}

        treated_level = _series(case["treated"], case["treated_source"])
        treated_pct = pct_series(treated_level, case["pre_months"])

        control_pcts = {}
        for region in case["controls"]:
            level = _read_datalab_level(region)
            control_pcts[region] = pct_series(level, case["pre_months"])

        def control_z_for(region_name, month):
            exclude = None if region_name == "__treated__" else region_name
            peers = [control_pcts[r][month] for r in case["controls"]
                     if r != exclude and month in control_pcts[r]]
            if len(peers) < 2:
                return np.nan
            mu, sd = np.mean(peers), np.std(peers, ddof=1)
            if not sd:
                return np.nan
            val = treated_pct[month] if region_name == "__treated__" else control_pcts[region_name][month]
            return (val - mu) / sd

        for m in case["post_months"]:
            if m in treated_pct:
                rows.append(dict(case_id=case_id, region=case["treated"], month=m,
                                  control_z=control_z_for("__treated__", m), label=1))
        for region in case["controls"]:
            for m in case["post_months"]:
                if m in control_pcts[region]:
                    rows.append(dict(case_id=case_id, region=region, month=m,
                                      control_z=control_z_for(region, m), label=0))
    return rows


def evaluate(rows):
    case_ids = list(CASES.keys())
    aucs = []
    for test_case in case_ids:
        train = [r for r in rows if r["case_id"] != test_case and np.isfinite(r["control_z"])]
        test = [r for r in rows if r["case_id"] == test_case and np.isfinite(r["control_z"])]
        X_train = np.array([[r["control_z"]] for r in train])
        y_train = np.array([r["label"] for r in train])
        X_test = np.array([[r["control_z"]] for r in test])
        y_test = np.array([r["label"] for r in test])

        model = LogisticRegression().fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        aucs.append(auc)
        print(f"  [테스트={test_case:15s}] 행수={len(test):2d}  AUC={auc:.2f}")
    print(f"\n  평균 LOCO AUC = {np.mean(aucs):.3f}  (기존 기준선 pct_change/전체기간=0.693 대비 "
          f"{np.mean(aucs)-0.693:+.3f})")
    return np.mean(aucs)


def main():
    rows = build_dataset()
    print(f"최종 데이터셋: {len(rows)}행 (치료군 {sum(r['label'] for r in rows)} / "
          f"대조군 {sum(1 - r['label'] for r in rows)})\n")
    print("피처: control_z (같은 달 대조군 대비 상대 표준편차)  |  창: 개봉후 4개월\n")
    evaluate(rows)

    # 전체 데이터로 최종 학습(실전 배포용) — 최종 모델 계수 확인
    valid = [r for r in rows if np.isfinite(r["control_z"])]
    X = np.array([[r["control_z"]] for r in valid])
    y = np.array([r["label"] for r in valid])
    final_model = LogisticRegression().fit(X, y)
    print(f"\n최종 배포용 모델(전체 데이터로 재학습): "
          f"logit(p) = {final_model.intercept_[0]:+.3f} + {final_model.coef_[0][0]:+.3f} * control_z")
    print("해석: control_z가 클수록(같은 대조군들보다 훨씬 더 튀는 지역-달일수록) "
          "'영화 효과로 인한 방문 급증'일 확률이 높다고 판단.")


if __name__ == "__main__":
    main()

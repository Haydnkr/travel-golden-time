"""
이상탐지/분류 모델 — "이 지역-이 달이 영화 효과로 튄 것인가?"를 맞히는 실제 학습 모델.

지금까지의 문제: 영화 "사례"가 3~4개뿐이라 회귀모델을 학습시킬 표본이 없었음.
이 스크립트의 발상 전환: 사례가 아니라 "지역-달"을 단위로 보면, 대조군(12개
지역) + 치료군(4개 지역)의 개봉후 관측치를 전부 모아 20~30개+ 행짜리 진짜
분류 데이터셋을 만들 수 있음. 각 행은 "그 지역이 자기 자신의 개봉전 평균
대비 그 달에 몇 % 달라졌나"(did_analysis.py의 레벨 기반 DID와 동일한 계산)
하나의 피처로 표현되고, 라벨은 "치료군 자신의 지역인가(1) 대조군인가(0)".

검증은 여전히 leave-one-case-out(사례 4개 중 3개로 학습, 1개로 검증) —
같은 사례 안의 여러 달은 서로 독립이 아니므로, 행 단위 랜덤분할은 안 됨.

사용법: python anomaly_classifier.py
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

CASES = {
    "yeongwol_2026": dict(
        treated="영월", treated_source="datalab", controls=["화천", "횡성", "인제"],
        pre_months=["202512", "202601"],
        post_months=["202602", "202603", "202604", "202605", "202606", "202607", "202608"],
    ),
    "yesan_2026": dict(
        treated="예산", treated_source="datalab", controls=["홍성", "청양", "서천"],
        pre_months=["202602", "202603"],
        post_months=["202604", "202605", "202606", "202607", "202608"],
    ),
    "paemyo_2024": dict(
        treated="daily_visits_muju.csv", treated_source="daily", controls=["진안", "장수", "임실"],
        pre_months=["202309", "202310", "202311", "202312", "202401"],
        post_months=["202402", "202403", "202404", "202405", "202406", "202407", "202408"],
    ),
    "jasaneobo_2021": dict(
        treated="daily_visits_sinan.csv", treated_source="daily", controls=["완도", "진도", "고흥"],
        pre_months=["202010", "202011", "202012", "202101", "202102"],
        post_months=["202103", "202104", "202105", "202106", "202107", "202108", "202109", "202110"],
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


def build_dataset():
    rows = []
    for case_id, case in CASES.items():
        if case["treated_source"] == "datalab":
            treated_level = _read_datalab_level(case["treated"])
        else:
            treated_level = _read_daily_aggregated(case["treated"])

        pre_avg = np.mean([treated_level[m] for m in case["pre_months"]])
        for m in case["post_months"]:
            if m in treated_level:
                pct = (treated_level[m] - pre_avg) / pre_avg * 100
                rows.append(dict(case_id=case_id, region="__treated__", month=m,
                                  pct_change=pct, label=1))

        for region in case["controls"]:
            level = _read_datalab_level(region)
            pre_avg_c = np.mean([level[m] for m in case["pre_months"] if m in level])
            for m in case["post_months"]:
                if m in level:
                    pct = (level[m] - pre_avg_c) / pre_avg_c * 100
                    rows.append(dict(case_id=case_id, region=region, month=m,
                                      pct_change=pct, label=0))
    return rows


def main():
    rows = build_dataset()
    case_ids = list(CASES.keys())

    print(f"전체 데이터셋: {len(rows)}행 (치료군 {sum(r['label'] for r in rows)}행 / "
          f"대조군 {sum(1 - r['label'] for r in rows)}행)\n")

    accs, aucs, baseline_accs = [], [], []
    for test_case in case_ids:
        train_rows = [r for r in rows if r["case_id"] != test_case]
        test_rows = [r for r in rows if r["case_id"] == test_case]

        X_train = np.array([[r["pct_change"]] for r in train_rows])
        y_train = np.array([r["label"] for r in train_rows])
        X_test = np.array([[r["pct_change"]] for r in test_rows])
        y_test = np.array([r["label"] for r in test_rows])

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, pred)
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")
        baseline_acc = 1 - y_test.mean()  # "항상 대조군(0)이라고 찍기"의 정확도

        accs.append(acc); aucs.append(auc); baseline_accs.append(baseline_acc)
        print(f"[테스트={test_case}] 행수={len(test_rows)}  정확도={acc:.2f}  "
              f"AUC={auc:.2f}  (항상-대조군-예측 기준선={baseline_acc:.2f})")

    print(f"\n평균 정확도: {np.mean(accs):.2f} (기준선 평균: {np.mean(baseline_accs):.2f})")
    print(f"평균 AUC: {np.nanmean(aucs):.2f} (0.5=무작위, 1.0=완벽)")


if __name__ == "__main__":
    main()

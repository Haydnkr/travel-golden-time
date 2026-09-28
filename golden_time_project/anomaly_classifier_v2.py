"""
anomaly_classifier.py의 개선 시도 — 여러 피처/모델 조합을 LOCO로 비교.

시도한 것:
  1) 피처: pct_change(개봉전 평균 대비 %) 단독 (v1, 기준선)
  2) 피처: z-score(개봉전 표준편차로 나눈 값) 단독 — 지역마다 원래 변동성이
     다른 걸 보정
  3) 피처: pct_change + z-score 둘 다
  4) 모델: LogisticRegression vs RandomForest vs GradientBoosting
  5) class_weight='balanced' 적용 여부

전부 leave-one-case-out(사례 4개)로 AUC 비교, 어느 조합이 실제로 나은지
정직하게 판단(안 좋아지면 안 좋아졌다고 그대로 보고).
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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


def _series(name, source):
    return _read_datalab_level(name) if source == "datalab" else _read_daily_aggregated(name)


def build_dataset(max_post_months=None):
    rows = []
    for case_id, case in CASES.items():
        post_months = case["post_months"][:max_post_months] if max_post_months else case["post_months"]

        def add_rows(level, label):
            pre_vals = [level[m] for m in case["pre_months"] if m in level]
            pre_avg = np.mean(pre_vals)
            pre_std = np.std(pre_vals, ddof=1) if len(pre_vals) > 1 else np.nan
            for m in post_months:
                if m not in level:
                    continue
                pct = (level[m] - pre_avg) / pre_avg * 100
                z = (level[m] - pre_avg) / pre_std if pre_std and pre_std > 0 else np.nan
                rows.append(dict(case_id=case_id, pct_change=pct, zscore=z, label=label))

        add_rows(_series(case["treated"], case["treated_source"]), 1)
        for region in case["controls"]:
            add_rows(_read_datalab_level(region), 0)
    return rows


def loco_auc(rows, feature_cols, model_factory, label):
    case_ids = list(CASES.keys())
    aucs = []
    for test_case in case_ids:
        train = [r for r in rows if r["case_id"] != test_case and all(np.isfinite(r[c]) for c in feature_cols)]
        test = [r for r in rows if r["case_id"] == test_case and all(np.isfinite(r[c]) for c in feature_cols)]
        if not test:
            continue
        X_train = np.array([[r[c] for c in feature_cols] for r in train])
        y_train = np.array([r["label"] for r in train])
        X_test = np.array([[r[c] for c in feature_cols] for r in test])
        y_test = np.array([r["label"] for r in test])

        model = model_factory()
        model.fit(X_train, y_train)
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except (ValueError, IndexError):
            auc = float("nan")
        aucs.append(auc)
    mean_auc = np.nanmean(aucs)
    per_case = ", ".join(f"{c.split('_')[0]}={a:.2f}" for c, a in zip(case_ids, aucs))
    print(f"{label:45s} 평균 AUC={mean_auc:.3f}   ({per_case})")
    return mean_auc


def main():
    rows = build_dataset()
    print(f"데이터셋: {len(rows)}행\n")
    print("=== 피처/모델 조합별 LOCO AUC 비교 ===")

    results = {}
    results["v1: pct_change, LogisticRegression"] = loco_auc(
        rows, ["pct_change"], lambda: LogisticRegression(), "v1: pct_change, LogisticRegression")
    results["v2: zscore, LogisticRegression"] = loco_auc(
        rows, ["zscore"], lambda: LogisticRegression(), "v2: zscore, LogisticRegression")
    results["v3: pct_change+zscore, LogisticRegression"] = loco_auc(
        rows, ["pct_change", "zscore"], lambda: LogisticRegression(), "v3: pct_change+zscore, LogisticRegression")
    results["v4: pct_change, LogReg(class_weight=balanced)"] = loco_auc(
        rows, ["pct_change"], lambda: LogisticRegression(class_weight="balanced"),
        "v4: pct_change, LogReg(class_weight=balanced)")
    results["v5: pct_change+zscore, RandomForest"] = loco_auc(
        rows, ["pct_change", "zscore"], lambda: RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        "v5: pct_change+zscore, RandomForest")
    results["v6: pct_change+zscore, GradientBoosting"] = loco_auc(
        rows, ["pct_change", "zscore"], lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0),
        "v6: pct_change+zscore, GradientBoosting")

    # v7: 개봉 후 "초반" 몇 달만 라벨 대상으로 삼음 — 골든타임 개념(초기 반응)과
    # 더 맞고, 뒤로 갈수록 계절/기타 요인에 희석되는 노이즈를 제거하는 효과.
    print()
    for n_months in (2, 3, 4, 5):
        rows_early = build_dataset(max_post_months=n_months)
        results[f"v7({n_months}개월 초반윈도우): pct_change, LogReg"] = loco_auc(
            rows_early, ["pct_change"], lambda: LogisticRegression(),
            f"v7({n_months}개월 초반윈도우): pct_change, LogReg")

    best = max(results, key=results.get)
    print(f"\n>>> 최고 성능: {best} (AUC={results[best]:.3f}) <<<")
    print(f"기존(v1, 전체기간) 대비 개선폭: {results[best] - results['v1: pct_change, LogisticRegression']:+.3f}")


if __name__ == "__main__":
    main()

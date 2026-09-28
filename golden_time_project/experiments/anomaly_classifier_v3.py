"""
anomaly_classifier_v2.py 다음 시도 — "기준선을 바꾸면 나아지는가?"

v1/v2까지 피처는 전부 "그 지역 자신의 개봉전 평균 대비 이번 달 몇 % 변했나"
(pct_change)였음. 문제: 사례마다 개봉전 기간 길이가 다르고(예산은 겨우 2개월치
데이터라 기준선이 불안정), 지역마다 원래 변동성 크기도 다름 — 그래서 사례를
넘나드는 학습(LOCO)이 잘 안 됐을 가능성.

이번에 시도하는 것: "같은 사례, 같은 달의 대조군들과 비교해서 이 지역이 얼마나
튀는가"(횡단면 비교, cross-sectional) 피처.
  control_z = (이 지역의 pct_change - 그 달 다른 대조군들의 평균)
              / 그 달 다른 대조군들의 표준편차
  (대조군 자기 자신을 뺀 "나머지 대조군"과 비교 — leave-one-out)

기존 pct_change는 "시간축 비교"(자기 과거 대비), 이건 "지역축 비교"(동시대
다른 지역 대비) — DID 아이디어를 다른 방식으로 한 번 더 적용한 것.

추가로 이 피처를 early-window(v7에서 효과 있었던 개봉 후 2개월 제한)와
결합했을 때도 시도.

전부 leave-one-case-out AUC로 정직하게 비교(안 좋으면 안 좋다고 보고).
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
    """각 행: pct_change(자기 개봉전 평균 대비, 기존), control_z(같은 달 다른
    대조군들 대비 얼마나 튀는지, 신규) 두 피처를 함께 계산."""
    rows = []
    for case_id, case in CASES.items():
        post_months = case["post_months"][:max_post_months] if max_post_months else case["post_months"]

        # 1) 지역별 pct_change 시계열(월 -> 값) 미리 계산
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

        # 2) 각 달마다: 대조군들의 pct_change 분포(leave-one-out)로 control_z 계산
        def control_z_for(region_name, month, exclude_self):
            peers = [control_pcts[r][month] for r in case["controls"]
                     if r != exclude_self and month in control_pcts[r]]
            if len(peers) < 2:
                return np.nan
            mu, sd = np.mean(peers), np.std(peers, ddof=1)
            if not sd or sd == 0:
                return np.nan
            val = treated_pct[month] if region_name == "__treated__" else control_pcts[region_name][month]
            return (val - mu) / sd

        for m in post_months:
            if m in treated_pct:
                rows.append(dict(
                    case_id=case_id, pct_change=treated_pct[m],
                    control_z=control_z_for("__treated__", m, exclude_self=None), label=1))
        for region in case["controls"]:
            for m in post_months:
                if m in control_pcts[region]:
                    rows.append(dict(
                        case_id=case_id, pct_change=control_pcts[region][m],
                        control_z=control_z_for(region, m, exclude_self=region), label=0))
    return rows


def loco_auc(rows, feature_cols, model_factory, label):
    case_ids = list(CASES.keys())
    aucs = []
    for test_case in case_ids:
        train = [r for r in rows if r["case_id"] != test_case and all(np.isfinite(r[c]) for c in feature_cols)]
        test = [r for r in rows if r["case_id"] == test_case and all(np.isfinite(r[c]) for c in feature_cols)]
        if not test:
            aucs.append(float("nan"))
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
    print(f"{label:50s} 평균 AUC={mean_auc:.3f}   ({per_case})")
    return mean_auc


def main():
    print("=== v8: 대조군-상대 z-score(횡단면 비교) 피처 실험 ===\n")

    results = {}

    rows_full = build_dataset()
    print(f"전체기간 데이터셋: {len(rows_full)}행\n")
    results["v1(기존 재확인): pct_change, 전체기간"] = loco_auc(
        rows_full, ["pct_change"], lambda: LogisticRegression(), "v1(기존 재확인): pct_change, 전체기간")
    results["v8: control_z 단독, 전체기간"] = loco_auc(
        rows_full, ["control_z"], lambda: LogisticRegression(), "v8: control_z 단독, 전체기간")
    results["v9: pct_change+control_z, 전체기간"] = loco_auc(
        rows_full, ["pct_change", "control_z"], lambda: LogisticRegression(), "v9: pct_change+control_z, 전체기간")

    print()
    for n_months in (2, 3, 4, 5):
        rows_early = build_dataset(max_post_months=n_months)
        results[f"v10({n_months}개월): control_z 단독"] = loco_auc(
            rows_early, ["control_z"], lambda: LogisticRegression(),
            f"v10({n_months}개월): control_z 단독")
        results[f"v11({n_months}개월): pct_change+control_z"] = loco_auc(
            rows_early, ["pct_change", "control_z"], lambda: LogisticRegression(),
            f"v11({n_months}개월): pct_change+control_z")

    baseline = results["v1(기존 재확인): pct_change, 전체기간"]
    prior_best = 0.812  # v7(2개월), 이전 스크립트 결과
    best = max(results, key=results.get)
    print(f"\n>>> 이번 실험 중 최고: {best} (AUC={results[best]:.3f}) <<<")
    print(f"기존 v1(전체기간 pct_change) 대비: {results[best] - baseline:+.3f}")
    print(f"지금까지 전체 최고 기록(v7, 2개월 pct_change 단독)={prior_best:.3f} 대비: "
          f"{results[best] - prior_best:+.3f}")


if __name__ == "__main__":
    main()

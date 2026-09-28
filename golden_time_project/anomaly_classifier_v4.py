"""
v3(control_z 피처)의 다음 시도.

v3에서 알게 된 것: control_z(같은 달 대조군 대비 상대 비교) 단독이 pct_change
단독보다 대체로 낫고(전체기간 0.74 vs 0.69), 어떤 개월수를 골라도 성능이
크게 안 흔들림(0.74~0.79 사이) — pct_change는 개월수에 따라 0.69~0.81로
더 들쭉날쭉했음. 근데 두 피처를 "그냥 합쳐서"(같은 모델에 같이 넣기) 쓰면
오히려 나빠짐(과적합, 사례 4개뿐이라 피처 2개짜리 결정경계를 안정적으로
못 배움) — v2에서도 봤던 패턴 반복.

이번에 시도하는 것:
  1) 앙상블: pct_change 모델과 control_z 모델을 "따로따로" 학습시키고,
     예측 확률만 평균 — 피처를 합치는 게 아니라 "두 전문가 의견을 평균"내는
     방식이라 차원이 안 늘어남 (과적합 위험이 훨씬 적음)
  2) 강한 정규화: 피처 2개를 같이 쓰되, 표준화 후 L2 정규화를 세게 걸어서
     (C=0.1, 0.01) 결정경계를 단순하게 강제 — 정말 정규화 부족이 원인이었는지 확인

전부 leave-one-case-out AUC로 비교.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from anomaly_classifier_v3 import CASES, build_dataset  # noqa: E402


def loco_ensemble_auc(rows, label):
    """pct_change 모델과 control_z 모델을 따로 학습, 예측 확률을 평균."""
    case_ids = list(CASES.keys())
    aucs = []
    for test_case in case_ids:
        train = [r for r in rows if r["case_id"] != test_case
                 and np.isfinite(r["pct_change"]) and np.isfinite(r["control_z"])]
        test = [r for r in rows if r["case_id"] == test_case
                and np.isfinite(r["pct_change"]) and np.isfinite(r["control_z"])]
        if not test:
            aucs.append(float("nan"))
            continue

        y_train = np.array([r["label"] for r in train])
        y_test = np.array([r["label"] for r in test])

        m1 = LogisticRegression().fit(np.array([[r["pct_change"]] for r in train]), y_train)
        m2 = LogisticRegression().fit(np.array([[r["control_z"]] for r in train]), y_train)

        p1 = m1.predict_proba(np.array([[r["pct_change"]] for r in test]))[:, 1]
        p2 = m2.predict_proba(np.array([[r["control_z"]] for r in test]))[:, 1]
        proba = (p1 + p2) / 2

        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
    mean_auc = np.nanmean(aucs)
    per_case = ", ".join(f"{c.split('_')[0]}={a:.2f}" for c, a in zip(case_ids, aucs))
    print(f"{label:45s} 평균 AUC={mean_auc:.3f}   ({per_case})")
    return mean_auc


def loco_regularized_auc(rows, C, label):
    """pct_change+control_z 표준화 후 강한 L2 정규화(C 작을수록 강함)."""
    case_ids = list(CASES.keys())
    aucs = []
    for test_case in case_ids:
        train = [r for r in rows if r["case_id"] != test_case
                 and np.isfinite(r["pct_change"]) and np.isfinite(r["control_z"])]
        test = [r for r in rows if r["case_id"] == test_case
                and np.isfinite(r["pct_change"]) and np.isfinite(r["control_z"])]
        if not test:
            aucs.append(float("nan"))
            continue

        X_train = np.array([[r["pct_change"], r["control_z"]] for r in train])
        y_train = np.array([r["label"] for r in train])
        X_test = np.array([[r["pct_change"], r["control_z"]] for r in test])
        y_test = np.array([r["label"] for r in test])

        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(C=C).fit(scaler.transform(X_train), y_train)
        try:
            proba = model.predict_proba(scaler.transform(X_test))[:, 1]
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
    mean_auc = np.nanmean(aucs)
    per_case = ", ".join(f"{c.split('_')[0]}={a:.2f}" for c, a in zip(case_ids, aucs))
    print(f"{label:45s} 평균 AUC={mean_auc:.3f}   ({per_case})")
    return mean_auc


def main():
    print("=== v12: 앙상블(확률 평균) vs 강한 정규화 실험 ===\n")
    results = {}

    for n_months in (None, 2, 3, 4, 5):
        rows = build_dataset(max_post_months=n_months)
        tag = "전체기간" if n_months is None else f"{n_months}개월"
        results[f"v12(앙상블,{tag})"] = loco_ensemble_auc(rows, f"v12(앙상블,{tag})")

    print()
    for n_months in (None, 2, 3, 4, 5):
        rows = build_dataset(max_post_months=n_months)
        tag = "전체기간" if n_months is None else f"{n_months}개월"
        for C in (0.1, 0.01):
            results[f"v13(정규화C={C},{tag})"] = loco_regularized_auc(rows, C, f"v13(정규화C={C},{tag})")

    prior_best = 0.812  # v7: 2개월, pct_change 단독
    v3_best = 0.786      # v3: 4개월, control_z 단독
    best = max(results, key=results.get)
    print(f"\n>>> 이번 실험 중 최고: {best} (AUC={results[best]:.3f}) <<<")
    print(f"기존 최고 기록들: v7(2개월,pct_change 단독)={prior_best:.3f}, "
          f"v10(4개월,control_z 단독)={v3_best:.3f}")
    overall_best_name, overall_best_val = ("v7(2개월,pct_change 단독)", prior_best)
    if v3_best > overall_best_val:
        overall_best_name, overall_best_val = ("v10(4개월,control_z 단독)", v3_best)
    if results[best] > overall_best_val:
        overall_best_name, overall_best_val = (best, results[best])
    print(f">>> 지금까지 전체 최고: {overall_best_name} (AUC={overall_best_val:.3f}) <<<")


if __name__ == "__main__":
    main()

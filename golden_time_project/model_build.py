"""
골든타임 예측모델 — 데이터 병합 + 학습 + 검증.

데이터가 절대적으로 부족한 상황(라벨 있는 사례가 영월/예산 2개뿐)에서
"학습시키는 것 만큼의 데이터는 없어도 쓸만한 예측모델"을 만들기 위한 접근:

  1. 사례(케이스) 2개뿐이라도, 그 안의 "일별 관측치"는 500개 넘게 있음
     → 일별 패널로 취급해서 일반적인 회귀/트리 모델 학습에 쓸 만한 표본 크기 확보.
  2. 단, 일별 관측치는 사례 내부에서 서로 강하게 상관돼 있어(같은 영화의
     연속된 날짜) 무작위로 학습/검증을 나누면 정보누출로 성능이 부풀려짐
     → 검증은 반드시 "사례 단위 leave-one-case-out"으로만 함
     (영월로 학습→예산으로 검증, 예산으로 학습→영월로 검증).
  3. 복잡한 모델이 단순한 규칙보다 실제로 나은지 항상 대조군(Plan B)과
     비교해서 확인 — 데이터가 적을수록 복잡한 모델이 오히려 과적합되기 쉬움.

Plan A: NAVER 검색지수 등 풍부한 피처 + 회귀/트리 모델 (팀원 데이터 기반, 영월·예산만 적용 가능)
Plan B: 검색지수 하나만 쓰는 단순회귀 — Plan A가 이것보다 못하면 Plan A는 기각
Plan C: Google Trends 검색지수만 사용(우리가 5개 사례 모두에서 구할 수 있는 유일한 신호)
        — Naver 대신 Google Trends를 써도 얼마나 성능이 유지되는지 확인.
        이게 잘 나오면 라벨 없는 곡성/국제시장/군위에도 나중에 적용 여지가 생김.

실행: python model_build.py
"""

import csv
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

TEAMMATE_XLSX = r"C:\Users\rabbi\Desktop\2026\School\competi\travel\golden_time_project\data\raw\영화관광_학습용_3개영화.xlsx"
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUT_CSV = os.path.join(RAW_DIR, "model_dataset_일별통합_병합.csv")


def load_teammate_panel():
    df = pd.read_excel(TEAMMATE_XLSX, sheet_name="일별통합")
    df["날짜"] = pd.to_datetime(df["날짜"])
    return df


def load_our_google_trends():
    """우리가 수집한 Google Trends 지역지수(영월/예산)를 날짜+지역 기준으로 병합할 수 있게 정리."""
    frames = []
    for fname, keycol, region in [
        ("googletrends_yeongwol_daily_partial.csv", "search_index_google_trends", "영월군"),
        ("googletrends_yeongwol_daily_part2.csv", "영월", "영월군"),
        ("googletrends_yesan_daily.csv", "예산", "예산군"),
        ("googletrends_gunwi_daily.csv", "군위", "군위군"),
        ("googletrends_littleforest_daegu_daily.csv", "리틀포레스트", "군위군"),
    ]:
        path = os.path.join(RAW_DIR, fname)
        # note 컬럼에 콤마가 섞여 있어 pandas C파서가 깨짐 -> csv.DictReader로 안전하게 읽음
        # (초과 필드는 DictReader가 알아서 무시 가능한 곳에 담아줌, 여기선 앞 2개 컬럼만 사용)
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                date_val = r.get("date")
                col = keycol if keycol in r else list(r.keys())[1]
                rows.append({"날짜": date_val, "GoogleTrends_지역지수": r.get(col)})
        out = pd.DataFrame(rows)
        out["날짜"] = pd.to_datetime(out["날짜"])
        out["지역"] = region
        frames.append(out)
    gt = pd.concat(frames, ignore_index=True)
    gt = gt.dropna(subset=["GoogleTrends_지역지수"])
    gt["GoogleTrends_지역지수"] = pd.to_numeric(gt["GoogleTrends_지역지수"], errors="coerce")
    # 중복 날짜(part1/part2 겹치는 구간)는 뒤 값 우선
    gt = gt.drop_duplicates(subset=["날짜", "지역"], keep="last")
    return gt


def build_features(df):
    """
    핵심 설계 결정: 타깃/피처를 절대값이 아니라 '직전 28일 평균 대비 비율'로 둠.
    영월·예산은 방문자 규모(스케일) 자체가 완전히 다른 지역이라, 절대 방문자수를
    직접 예측하면 한쪽 지역으로 학습한 모델이 다른 지역엔 스케일이 안 맞아
    전혀 못 맞춘다(실제로 1차 시도에서 R^2가 크게 음수로 나와 확인됨).
    비율로 바꾸면 "평시 대비 몇 배"라는 스케일 독립적인 값이 되어 두 지역 간
    모델 전이가 비로소 말이 되게 됨. 팀원이 이미 NAVER_28일기준지수·
    방문지수_28일기준을 이런 방식으로 설계해둔 것과 같은 원리.
    """
    df = df.sort_values(["지역", "날짜"]).reset_index(drop=True)

    # 우리 Google Trends도 같은 방식(직전 28일 평균 대비 비율)으로 정규화
    df["GoogleTrends_직전28일평균"] = (
        df.groupby("지역")["GoogleTrends_지역지수"]
        .transform(lambda s: s.shift(1).rolling(28, min_periods=10).mean())
    )
    df["GoogleTrends_28일기준비율"] = df["GoogleTrends_지역지수"] / df["GoogleTrends_직전28일평균"]

    for lag in (1, 3, 7):
        df[f"NAVER당일28일비율_lag{lag}"] = df.groupby("지역")["NAVER_당일28일비율"].shift(lag)
        df[f"GoogleTrends28일비율_lag{lag}"] = df.groupby("지역")["GoogleTrends_28일기준비율"].shift(lag)

    df["개봉후경과일_로그"] = np.sign(df["개봉후경과일"]) * np.log1p(df["개봉후경과일"].abs())
    # 타깃: 전체방문자의 직전28일 평균 대비 비율(팀원 산출식, 1=평시수준) → 로그변환해 대칭화
    df["방문지수_로그"] = np.log(df["방문지수_28일기준"].replace(0, np.nan))
    return df


def loco_eval(df, feature_cols, target_col="방문지수_로그", model_factory=None, label=""):
    """leave-one-case-out: 지역 하나로 학습, 나머지 하나로 검증 (2개뿐이라 2-fold).
    타깃이 로그(비율)이므로 MAE는 exp()해서 '평시 대비 배수' 오차로 환산해서 같이 보여줌."""
    regions = df["지역"].unique()
    rows = []
    for test_region in regions:
        train = df[df["지역"] != test_region].dropna(subset=feature_cols + [target_col])
        test = df[df["지역"] == test_region].dropna(subset=feature_cols + [target_col])
        if len(train) < 20 or len(test) < 20:
            continue
        model = model_factory()
        model.fit(train[feature_cols], train[target_col])
        pred_log = model.predict(test[feature_cols])
        actual_log = test[target_col].values
        mae_log = mean_absolute_error(actual_log, pred_log)
        r2 = r2_score(actual_log, pred_log)
        # 로그 오차를 "평시 대비 배수 오차"로 환산 (예: 0.3 -> 실제 대비 약 1.35배 정도 어긋남)
        mae_ratio = np.expm1(mae_log)
        rows.append({"plan": label, "test_region": test_region,
                      "n_train": len(train), "n_test": len(test),
                      "MAE(배수오차)": round(mae_ratio, 3), "R2(log비율)": round(r2, 3)})
    return pd.DataFrame(rows)


def main():
    print("1) 팀원 데이터 로드...")
    team = load_teammate_panel()
    print(f"   팀원 일별통합: {team.shape[0]}행, 지역={team['지역'].unique().tolist()}")

    print("2) 우리 Google Trends 데이터 로드 및 병합...")
    gt = load_our_google_trends()
    merged = team.merge(gt, on=["날짜", "지역"], how="left")
    print(f"   병합 후: {merged.shape[0]}행, GoogleTrends 결측률={merged['GoogleTrends_지역지수'].isna().mean():.1%}")

    print("3) 피처 엔지니어링...")
    merged = build_features(merged)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"   저장: {OUT_CSV}")

    print("\n4) Plan B (베이스라인, NAVER 당일/28일비율 단독)")
    plan_b = loco_eval(
        merged, ["NAVER_당일28일비율"],
        model_factory=lambda: Ridge(alpha=1.0), label="B_naver비율단독_Ridge",
    )
    print(plan_b.to_string(index=False))

    print("\n4-1) Plan A-0 (달력 정보만, 검색 신호 전혀 없이 — 검색이 실제로 뭘 더해주는지 확인용 대조군)")
    feat_a0 = ["주말", "개봉후경과일_로그"]
    plan_a0 = loco_eval(merged, feat_a0, model_factory=lambda: Ridge(alpha=1.0), label="A0_달력만_Ridge")
    print(plan_a0.to_string(index=False))

    print("\n5) Plan A-1 (Ridge, 풍부한 NAVER 비율 파생 피처)")
    feat_a = ["NAVER_당일28일비율", "NAVER당일28일비율_lag1", "NAVER당일28일비율_lag3",
              "NAVER당일28일비율_lag7", "NAVER_28일기준지수", "주말", "개봉후경과일_로그"]
    plan_a1 = loco_eval(merged, feat_a, model_factory=lambda: Ridge(alpha=1.0), label="A1_Ridge")
    print(plan_a1.to_string(index=False))

    print("\n6) Plan A-2 (GradientBoosting, 동일 피처)")
    plan_a2 = loco_eval(
        merged, feat_a,
        model_factory=lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0
        ),
        label="A2_GBR",
    )
    print(plan_a2.to_string(index=False))

    print("\n7) Plan C (Google Trends 비율 단독 — 5개 사례 전부에 적용 가능한 신호로 대체했을 때 성능)")
    feat_c = ["GoogleTrends_28일기준비율", "GoogleTrends28일비율_lag1",
              "GoogleTrends28일비율_lag3", "GoogleTrends28일비율_lag7"]
    plan_c = loco_eval(merged, feat_c, model_factory=lambda: Ridge(alpha=1.0), label="C_GoogleTrends비율단독_Ridge")
    print(plan_c.to_string(index=False))

    print("\n=== 요약 (지역별 평균 성능) ===")
    all_results = pd.concat([plan_a0, plan_b, plan_a1, plan_a2, plan_c], ignore_index=True)
    summary = all_results.groupby("plan")[["MAE(배수오차)", "R2(log비율)"]].mean().round(3)
    print(summary.to_string())
    summary_path = os.path.join(RAW_DIR, "model_loco_results.csv")
    all_results.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {summary_path}")

    print("\n7-1) *** 중요 발견: A0(달력만) vs A1(달력+검색) 성능이 거의 동일 ***")
    print("    -> '오늘 검색이 오늘 방문에 오늘 얼마나 몰리는지'를 맞히는 문제로는")
    print("       검색 신호가 요일 정보 이상의 추가 정보를 거의 안 줌.")
    print("       골든타임의 핵심 주장('검색이 방문보다 먼저 온다')을 검증하려면")
    print("       같은 날을 맞히는 게 아니라 '오늘 검색으로 며칠 뒤 방문을 맞히는지'를")
    print("       봐야 함 -> Plan A-3으로 재구성해서 확인.")

    print("\n7-2) Plan A-3 (진짜 예측: 오늘 검색 -> 7일 뒤 방문지수 예측)")
    horizon = 7
    fdf = merged.copy()
    fdf["방문지수_로그_h7"] = fdf.groupby("지역")["방문지수_로그"].shift(-horizon)
    fdf["주말_h7"] = fdf.groupby("지역")["주말"].shift(-horizon)
    fdf["개봉후경과일_로그_h7"] = fdf.groupby("지역")["개봉후경과일_로그"].shift(-horizon)
    feat_a3_calendar_only = ["주말_h7", "개봉후경과일_로그_h7"]
    feat_a3_full = feat_a3_calendar_only + ["NAVER_당일28일비율", "NAVER당일28일비율_lag1",
                                             "NAVER당일28일비율_lag3", "NAVER당일28일비율_lag7",
                                             "NAVER_28일기준지수"]
    plan_a3_cal = loco_eval(fdf, feat_a3_calendar_only, target_col="방문지수_로그_h7",
                             model_factory=lambda: Ridge(alpha=1.0), label="A3_h7_달력만")
    plan_a3_full = loco_eval(fdf, feat_a3_full, target_col="방문지수_로그_h7",
                              model_factory=lambda: Ridge(alpha=1.0), label="A3_h7_달력+검색")
    print("  [달력만, 7일뒤 예측]")
    print(plan_a3_cal.to_string(index=False))
    print("  [달력+검색, 7일뒤 예측]")
    print(plan_a3_full.to_string(index=False))
    all_results = pd.concat([all_results, plan_a3_cal, plan_a3_full], ignore_index=True)
    all_results.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n8) 최종 모델 학습 (A1_Ridge가 LOCO에서 제일 나음 -> 전체 데이터로 재학습)")
    final_data = merged.dropna(subset=feat_a + ["방문지수_로그"])
    final_model = Ridge(alpha=1.0)
    final_model.fit(final_data[feat_a], final_data["방문지수_로그"])
    coef_table = pd.Series(final_model.coef_, index=feat_a).sort_values(key=abs, ascending=False)
    print("피처별 계수(로그 스케일, 절대값 큰 순):")
    print(coef_table.round(3).to_string())

    import joblib
    model_dir = os.path.join(os.path.dirname(__file__), "model_artifacts")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "golden_time_ridge_v1.joblib")
    joblib.dump({"model": final_model, "features": feat_a, "target": "방문지수_로그(=log(당일 전체방문자/직전28일평균))"}, model_path)
    print(f"최종 모델 저장: {model_path}")
    print(f"학습에 쓰인 행 수: {len(final_data)} (영월/예산 합산, 결측 제외)")


if __name__ == "__main__":
    main()

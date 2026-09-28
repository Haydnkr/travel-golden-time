"""
이중차분법(Difference-in-Differences, DID) 분석.

golden_time_calculator.py의 한계: "검색량/방문자수가 평시보다 늘었다"만 보여줄 뿐,
그게 진짜 영화 때문인지 아니면 그 시기 전국적으로 다 늘어난 건지(계절, 명절,
여행 붐 등) 구분하지 못함. DID는 영화와 무관한 대조군(비슷한 규모의 다른 군)을
같이 봐서, "대상 지역의 변화폭 - 대조군의 변화폭" = 영화만의 순수 효과를 뽑아냄.

방법: 각 지역의 "전년동월방문자수증감률"(데이터랩이 이미 계산해주는 값,
계절성이 자동으로 제거된 지표)을 쓴다. 이 증감률의 개봉전 평균 대비 개봉후
평균 상승폭을, 대상 지역과 대조군 각각에 대해 구하고 그 차이를 DID 효과로 본다.

DID_효과 = (대상_개봉후평균YoY - 대상_개봉전평균YoY) - (대조군_개봉후평균YoY - 대조군_개봉전평균YoY)

사용법:
    python did_analysis.py                # 전체 사례
    python did_analysis.py --case yeongwol_2026
"""

import argparse
import csv
import os
import statistics
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def _read_visits(region):
    path = os.path.join(RAW_DIR, f"datalab_visits_{region}.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["기준년월"]: float(r["방문자수증감률"]) for r in rows}


CASES = {
    "yeongwol_2026": dict(
        treated="영월", controls=["화천", "횡성", "인제"],
        pre_months=["202512", "202601"],
        post_months=["202602", "202603", "202604", "202605", "202606", "202607", "202608"],
    ),
    "yesan_2026": dict(
        treated="예산", controls=["홍성", "청양", "서천"],
        pre_months=["202602", "202603"],
        post_months=["202604", "202605", "202606", "202607", "202608"],
    ),
}


def _avg(yoy_by_month, months):
    vals = [yoy_by_month[m] for m in months if m in yoy_by_month]
    if not vals:
        return None
    return statistics.mean(vals)


def run_did(case_id):
    case = CASES[case_id]
    treated_yoy = _read_visits(case["treated"])
    t_pre = _avg(treated_yoy, case["pre_months"])
    t_post = _avg(treated_yoy, case["post_months"])
    t_delta = t_post - t_pre

    print(f"\n=== {case_id} ({case['treated']}) ===")
    print(f"  대상({case['treated']}) 개봉전 평균 YoY 증감률: {t_pre:+.1f}%")
    print(f"  대상({case['treated']}) 개봉후 평균 YoY 증감률: {t_post:+.1f}%")
    print(f"  대상 자체 변화폭: {t_delta:+.1f}%p")

    control_deltas = []
    for c in case["controls"]:
        c_yoy = _read_visits(c)
        c_pre = _avg(c_yoy, case["pre_months"])
        c_post = _avg(c_yoy, case["post_months"])
        c_delta = c_post - c_pre
        control_deltas.append(c_delta)
        print(f"  대조군({c}) 개봉전 {c_pre:+.1f}% -> 개봉후 {c_post:+.1f}%  (변화폭 {c_delta:+.1f}%p)")

    control_delta_avg = statistics.mean(control_deltas)
    did_effect = t_delta - control_delta_avg

    print(f"  대조군 평균 변화폭: {control_delta_avg:+.1f}%p")
    print(f"  >>> DID 효과(순수 영화 효과 추정치): {did_effect:+.1f}%p <<<")
    print(f"  (대조군별 변화폭 범위: {min(control_deltas):+.1f}%p ~ {max(control_deltas):+.1f}%p "
          f"— 대조군마다 결과가 이 범위 안에서 흔들린다는 뜻, 표본 3개라 통계적 유의성 검정은 아님)")

    return dict(case_id=case_id, treated_delta=t_delta, control_delta_avg=control_delta_avg,
                did_effect=did_effect, control_deltas=control_deltas)


# ---------------------------------------------------------------------------
# 레벨 기반 DID (파묘·자산어보용) — 위 YoY 방식과 다른 이유:
# 이 두 사례는 대조군 기간(2020-2021년)이 데이터랩 시작(2020-01)과 겹쳐서
# "전년동월방문자수"가 없는 달이 있음. 그래서 "전년 동월 대비"가 아니라
# "자기 자신의 개봉전 평균 대비 몇 % 변했나"로 계산 방식만 바꿈 — 논리는 동일
# (대상의 변화율 - 대조군 평균 변화율 = 순수 효과).
LEVEL_CASES = {
    "paemyo_2024": dict(
        treated_daily_file="daily_visits_muju.csv",
        controls=["진안", "장수", "임실"],
        pre_months=["202309", "202310", "202311", "202312", "202401"],
        # 202308은 6일치뿐인 반쪽 달이라 제외(위 daily_visits_muju.csv 확인 결과)
        post_months=["202402", "202403", "202404", "202405", "202406", "202407", "202408"],
    ),
    "jasaneobo_2021": dict(
        treated_daily_file="daily_visits_sinan.csv",
        controls=["완도", "진도", "고흥"],
        pre_months=["202010", "202011", "202012", "202101", "202102"],
        post_months=["202103", "202104", "202105", "202106", "202107", "202108", "202109", "202110"],
    ),
}


def _aggregate_daily_to_monthly(fname, col="외지인방문자", min_days=20):
    """daily_visits_*.csv(팀원 제공)의 특정 컬럼을 월별 합계로 집계.
    데이터가 min_days일 미만인 달(월초/월말 반쪽 달)은 비교가 왜곡되니 제외."""
    path = os.path.join(RAW_DIR, fname)
    counts = {}
    sums = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = r.get(col)
            if not v:
                continue
            ym = r["date"][:7].replace("-", "")
            sums[ym] = sums.get(ym, 0.0) + float(v)
            counts[ym] = counts.get(ym, 0) + 1
    return {ym: total for ym, total in sums.items() if counts[ym] >= min_days}


def _read_visits_level(region):
    """datalab_visits_*.csv(대조군, 관광데이터랩 원본)의 원본 방문자수(외지인 기준)."""
    path = os.path.join(RAW_DIR, f"datalab_visits_{region}.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["기준년월"]: float(r["방문자수"]) for r in rows}


def run_did_level(case_id):
    case = LEVEL_CASES[case_id]
    treated_level = _aggregate_daily_to_monthly(case["treated_daily_file"])
    t_pre = _avg(treated_level, case["pre_months"])
    t_post = _avg(treated_level, case["post_months"])
    t_pct = (t_post - t_pre) / t_pre * 100

    print(f"\n=== {case_id} (레벨 기반 DID) ===")
    print(f"  대상 개봉전 평균 방문자(월,외지인): {t_pre:,.0f}명")
    print(f"  대상 개봉후 평균 방문자(월,외지인): {t_post:,.0f}명")
    print(f"  대상 자체 변화율: {t_pct:+.1f}%")

    control_pcts = []
    for c in case["controls"]:
        c_level = _read_visits_level(c)
        c_pre = _avg(c_level, case["pre_months"])
        c_post = _avg(c_level, case["post_months"])
        c_pct = (c_post - c_pre) / c_pre * 100
        control_pcts.append(c_pct)
        print(f"  대조군({c}) 개봉전 {c_pre:,.0f}명 -> 개봉후 {c_post:,.0f}명  (변화율 {c_pct:+.1f}%)")

    control_pct_avg = statistics.mean(control_pcts)
    did_effect = t_pct - control_pct_avg

    print(f"  대조군 평균 변화율: {control_pct_avg:+.1f}%")
    print(f"  >>> DID 효과(순수 영화 효과 추정치): {did_effect:+.1f}%p <<<")
    print(f"  (대조군별 변화율 범위: {min(control_pcts):+.1f}% ~ {max(control_pcts):+.1f}%)")

    return dict(case_id=case_id, treated_pct=t_pct, control_pct_avg=control_pct_avg,
                did_effect=did_effect, control_pcts=control_pcts)


def main():
    parser = argparse.ArgumentParser(description="이중차분법(DID) 분석기")
    parser.add_argument("--case", help="특정 case_id만 계산 (안 주면 전체)")
    args = parser.parse_args()

    if args.case:
        if args.case in CASES:
            run_did(args.case)
        elif args.case in LEVEL_CASES:
            run_did_level(args.case)
        else:
            raise SystemExit(f"알 수 없는 case_id: {args.case}")
        return

    for case_id in CASES:
        run_did(case_id)
    for case_id in LEVEL_CASES:
        run_did_level(case_id)


if __name__ == "__main__":
    main()

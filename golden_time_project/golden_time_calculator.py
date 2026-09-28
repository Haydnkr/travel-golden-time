"""
골든타임 계산기 — "새 데이터가 들어오면 바로 골든타임 갭을 계산"해주는 도구.

model_build.py에서 확인한 것: 사례가 2개뿐인 상황에서 회귀모델을 "학습"시켜도
요일 정보 이상을 못 맞힘. 그래서 이 도구는 학습된 모델이 아니라, 기존
analysis.py/preprocess.py의 통계적 탐지 로직(평시 대비 표준편차 초과)을
검색량과 박스오피스(흥행) 양쪽에 그대로 적용하는 방식으로 만들었음.
재학습이 필요 없고, 새 원자료 CSV만 CASES 아래 등록하면 바로 계산됨.

시작점(골든타임의 트리거)을 검색량과 박스오피스 두 신호로 각각 계산해서
같이 보여줌 — "영화 흥행 자체"와 "검색 관심" 중 어느 게 더 먼저/뚜렷하게
움직이는지 사례별로 다를 수 있어서, 하나로 뭉개지 않고 둘 다 계산함.
config.GOLDEN_TIME_START_DEF에 따라 최종 "공식 시작점"만 하나 고름.

사용법:
    python golden_time_calculator.py                    # 전체 사례
    python golden_time_calculator.py --case yeongwol_2026   # 사례 하나만

새 사례(예: 리틀포레스트 방문자수 확보됨) 추가하는 법:
    아래 CASES 딕셔너리에 항목 하나 추가 (release_date, search 파일들,
    boxoffice 파일, visits 파일, official_response_date) 하고 재실행.
    코드 수정 불필요, 재학습 불필요.
"""

import argparse
import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import preprocess
import analysis

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def _read_csv_safe(path, encoding="utf-8-sig"):
    """note 컬럼에 콤마가 섞인 CSV도 안전하게 읽음 (csv.DictReader는 초과
    필드를 알아서 처리해줌 — pandas C파서는 이런 파일에서 에러를 냄)."""
    with open(path, encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


def _load_search_series(files_and_cols):
    """[(파일명, 값컬럼명 또는 (긴포맷값컬럼,키워드컬럼)), ...] -> [date, search_index] DataFrame.
    여러 파일에 걸쳐 있으면 날짜 기준으로 병합(겹치면 뒤 파일 값 우선)."""
    merged = {}
    for fname, col_spec in files_and_cols:
        rows = _read_csv_safe(os.path.join(RAW_DIR, fname))
        for r in rows:
            d = r.get("date")
            if not d:
                continue
            if isinstance(col_spec, tuple):  # 긴 포맷: (값컬럼, 키워드컬럼)
                val = r.get(col_spec[0])
            else:
                val = r.get(col_spec)
            if val is None or val == "":
                continue
            merged[d] = float(val)
    if not merged:
        return pd.DataFrame(columns=["date", "search_index"])
    out = pd.DataFrame(sorted(merged.items()), columns=["date", "search_index"])
    out["date"] = pd.to_datetime(out["date"])
    return out


def _load_boxoffice_series(fname, fmt):
    """boxoffice_manual_*.csv -> [date, daily_audience] DataFrame.
    fmt='yeongwol'(일별 단일 컬럼) 또는 'typed'(date_or_range/type 컬럼 포함,
    daily/weekend_total/cumulative 혼재 — daily 타입만 골라서 씀. 예산·곡성·
    국제시장 파일이 모두 이 포맷)."""
    if not fname:
        return pd.DataFrame(columns=["date", "daily_audience"])
    rows = _read_csv_safe(os.path.join(RAW_DIR, fname))
    out = []
    for r in rows:
        if fmt == "yeongwol":
            out.append({"date": r["date"], "daily_audience": float(r["daily_audience"])})
        elif fmt == "typed":
            if r["type"] == "daily":
                out.append({"date": r["date_or_range"].split("~")[0],
                             "daily_audience": float(r["audience"])})
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["date", "daily_audience"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_visits_series(fname):
    """datalab_visits_*.csv(월별 방문자수) -> 일별 [date, visitor_count]로 확장.
    ⚠️ 월값을 그 달의 모든 날짜에 반복해서 채우는 것뿐, 진짜 일별 데이터가
    아님 — 방문 '급증일'을 일 단위로 정확히 짚는 용도로는 못 쓰고, 월 단위
    추세 확인/방문 피크 '월' 파악 정도의 참고용으로만 쓸 것."""
    if not fname:
        return pd.DataFrame(columns=["date", "visitor_count"])
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "visitor_count"])
    rows = _read_csv_safe(path)
    out = []
    for r in rows:
        yyyymm = r["기준년월"]
        year, month = int(yyyymm[:4]), int(yyyymm[4:6])
        days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
        for day in range(1, days_in_month + 1):
            out.append({"date": pd.Timestamp(year=year, month=month, day=day),
                         "visitor_count": float(r["방문자수"])})
    return pd.DataFrame(out)


def _load_visits_series_daily(fname):
    """daily_visits_*.csv(팀원 제공, convert_teammate_daily.py 산출물) -> [date, visitor_count].
    _load_visits_series()와 달리 월값 반복 확장이 아니라 진짜 일별 데이터 그대로 사용 —
    방문 '급증일'을 일 단위로 정확히 짚을 수 있음(전체방문자 기준)."""
    if not fname:
        return pd.DataFrame(columns=["date", "visitor_count"])
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "visitor_count"])
    rows = _read_csv_safe(path)
    out = []
    for r in rows:
        v = r.get("전체방문자")
        if v in (None, ""):
            continue
        out.append({"date": pd.Timestamp(r["date"]), "visitor_count": float(v)})
    return pd.DataFrame(out)


def _load_boxoffice_series_kobis_daily(fname):
    """daily_kobis_*.csv(팀원 제공, KOBIS 공식 일별 관객수) -> [date, daily_audience]."""
    if not fname:
        return pd.DataFrame(columns=["date", "daily_audience"])
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "daily_audience"])
    rows = _read_csv_safe(path)
    out = []
    for r in rows:
        v = r.get("KOBIS_관객수")
        if v in (None, ""):
            continue
        out.append({"date": r["date"], "daily_audience": float(v)})
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["date", "daily_audience"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# 사례별 원자료 등록 — 새 데이터 생기면 여기 항목만 추가/수정
CASES = {
    "yeongwol_2026": dict(
        content_title="왕과 사는 남자", region="영월",
        release_date="2026-02-04", official_response_date="2026-03-06",
        search_files=[
            ("googletrends_yeongwol_daily_partial.csv", "search_index_google_trends"),
            ("googletrends_yeongwol_daily_part2.csv", "영월"),
        ],
        boxoffice_file="boxoffice_manual_yeongwol_2026.csv", boxoffice_format="yeongwol",
        visits_file="datalab_visits_영월.csv",
        # 2026-09-27: 팀원 파일에서 진짜 일별 방문자 데이터 확보 — 이쪽을 우선 사용.
        visits_file_daily="daily_visits_yeongwol.csv",
    ),
    "yesan_2026": dict(
        content_title="살목지", region="예산",
        release_date="2026-04-08", official_response_date="2026-04-14",
        search_files=[("googletrends_yesan_daily.csv", "예산")],
        boxoffice_file="boxoffice_manual_yesan_2026.csv", boxoffice_format="typed",
        visits_file="datalab_visits_예산.csv",
        visits_file_daily="daily_visits_yesan.csv",
        boxoffice_file_daily="daily_kobis_yesan.csv",
    ),
    "gokseong_2016": dict(
        content_title="곡성", region="곡성",
        release_date="2016-05-12", official_response_date=None,
        search_files=[("googletrends_gokseong_daily.csv", "곡성")],
        boxoffice_file="boxoffice_manual_gokseong_2016.csv", boxoffice_format="typed",
        visits_file=None,
    ),
    "busan_2014": dict(
        content_title="국제시장", region="국제시장",
        release_date="2014-12-17", official_response_date=None,
        search_files=[("googletrends_gukjesijang_daily.csv", "국제시장")],
        boxoffice_file="boxoffice_manual_gukjesijang_2014.csv", boxoffice_format="typed",
        visits_file=None,
    ),
    "mungyeong_2026": dict(
        content_title="왕과 사는 남자", region="문경새재",
        release_date="2026-02-04", official_response_date="2026-03-14",
        search_files=[("googletrends_mungyeong_daily.csv", "문경")],
        # ⚠️ "문경새재"(고유 촬영지명) 단독 검색은 결측뿐이라 상위 지명 "문경"으로 대체 수집함.
        # "문경"은 일반 지명이라 영화와 무관한 배경 검색이 섞여 있을 수 있음 — 해석에 유의.
        boxoffice_file=None, boxoffice_format=None, visits_file=None,
    ),
    "sopoong_2024": dict(
        content_title="소풍", region="남해",
        release_date="2024-02-07", official_response_date=None,
        search_files=[("googletrends_sopoong_namhae_daily.csv", "소풍")],
        # ⚠️ "소풍"은 '피크닉'이라는 일반명사와 겹침 — 노이즈 있으나 개봉일 기점 상승은 확인됨.
        boxoffice_file=None, boxoffice_format=None, visits_file=None,
        visits_file_daily="daily_visits_namhae.csv", boxoffice_file_daily="daily_kobis_namhae.csv",
    ),
    "paemyo_2024": dict(
        content_title="파묘", region="무주",
        release_date="2024-02-22", official_response_date=None,
        search_files=[("googletrends_paemyo_muju_daily.csv", "파묘")],
        boxoffice_file=None, boxoffice_format=None, visits_file=None,
        visits_file_daily="daily_visits_muju.csv", boxoffice_file_daily="daily_kobis_muju.csv",
    ),
    "jasaneobo_2021": dict(
        content_title="자산어보", region="신안",
        release_date="2021-03-31", official_response_date=None,
        search_files=[("googletrends_jasaneobo_sinan_daily.csv", "자산어보")],
        boxoffice_file=None, boxoffice_format=None, visits_file=None,
        visits_file_daily="daily_visits_sinan.csv", boxoffice_file_daily="daily_kobis_sinan.csv",
    ),
    "haenam_2026": dict(
        content_title="호프", region="해남",
        release_date="2026-07-15", official_response_date=None,  # 미확인(사전기획형이라 성격이 다름 — 대화 참고)
        search_files=[],  # TODO: 구글 트렌드 아직 미저장(라이브 확인만 함) — 재수집 필요
        boxoffice_file=None, boxoffice_format=None, visits_file=None,
        visits_file_daily="daily_visits_haenam.csv", boxoffice_file_daily="daily_kobis_haenam.csv",
    ),
    "gunwi_2018": dict(
        content_title="리틀 포레스트", region="군위",
        release_date="2018-02-28", official_response_date=None,
        search_files=[("googletrends_gunwi_daily.csv", "군위"),
                       ("googletrends_littleforest_daegu_daily.csv", "리틀포레스트")],
        boxoffice_file=None, boxoffice_format=None, visits_file=None,
        # 2026-09-27: 팀원 파일에서 진짜 일별 방문자+KOBIS 확보 — test-set 격하 결정 재검토 필요.
        visits_file_daily="daily_visits_gunwi.csv", boxoffice_file_daily="daily_kobis_gunwi.csv",
    ),
}


def calculate_golden_time(case_id: str, threshold_multiplier: float = None, baseline_days: int = None,
                           boxoffice_multiplier: float = 1.5):
    case_files = CASES[case_id]
    case = {
        "case_id": case_id,
        "content_title": case_files["content_title"],
        "release_date": pd.Timestamp(case_files["release_date"]),
        "official_response_date": (
            pd.Timestamp(case_files["official_response_date"])
            if case_files["official_response_date"] else None
        ),
    }

    search_df = _load_search_series(case_files["search_files"])

    # 팀원 제공 일별 데이터(daily_kobis_*/daily_visits_*)가 있으면 그걸 우선 사용 —
    # 진짜 일별값이라 기존 월별-반복/수동수집 데이터보다 정확함.
    if case_files.get("boxoffice_file_daily"):
        box_df = _load_boxoffice_series_kobis_daily(case_files["boxoffice_file_daily"])
    else:
        box_df = _load_boxoffice_series(case_files["boxoffice_file"], case_files["boxoffice_format"])

    if case_files.get("visits_file_daily"):
        visits_df = _load_visits_series_daily(case_files["visits_file_daily"])
    else:
        visits_df = _load_visits_series(case_files["visits_file"])

    panel = preprocess.build_event_aligned_panel(case, box_df, search_df, visits_df)

    report = {"case_id": case_id, "content_title": case["content_title"]}

    # --- 검색량 기반 급증일 ---
    if not search_df.empty:
        panel_s = preprocess.compute_baseline_excess(panel, "search_index", baseline_days)
        search_spike = analysis.detect_search_spike_date(
            panel_s, value_col="search_index_excess", threshold_multiplier=threshold_multiplier
        )
        report["검색량_급증일"] = search_spike
    else:
        report["검색량_급증일"] = None
        report["검색량_참고"] = "검색량 데이터 없음"

    # --- 박스오피스(흥행) 기반 급증일 ---
    # 검색량과 다른 기준 사용: "평시(개봉전) 평균 대비"가 아니라 "개봉일 대비 배수".
    # 관객수는 개봉 전엔 정의상 0이라 평시 개념이 없고, 또 조기경보 취지상
    # "정점"이 아니라 "개봉일보다 늘어나는 추세로 돌아서는 날"을 봐야 함
    # (analysis.detect_boxoffice_momentum_date 참고).
    if not box_df.empty:
        box_spike = analysis.detect_boxoffice_momentum_date(
            panel, value_col="daily_audience", multiplier=boxoffice_multiplier,
        )
        report["흥행(관객수)_급증일"] = box_spike
    else:
        report["흥행(관객수)_급증일"] = None
        report["흥행_참고"] = "박스오피스 데이터 없음"

    # --- 종합 시작점: 검색/흥행 중 더 이른 날짜 (둘 다 있으면) ---
    candidates = [d for d in [report["검색량_급증일"], report["흥행(관객수)_급증일"]] if d is not None]
    trigger_date = min(candidates) if candidates else None
    report["종합_트리거일"] = trigger_date

    # --- 종료점: 공식대응일 우선, 없으면 방문자수 피크일(월 단위 주의) ---
    end_date = case["official_response_date"]
    end_date_note = "지자체 공식대응일"
    if end_date is None and not visits_df.empty:
        # 개봉 이후 구간에서만 피크를 찾음 — 아니면 그 지역의 원래 성수기(여름휴가철 등)가
        # 영화와 무관하게 "피크"로 잡혀버림(특히 방문자수가 원래 계절변동이 큰 관광지일 때).
        post_release = panel[panel["date"] >= case["release_date"]]
        peak_row = (post_release.loc[post_release["visitor_count"].idxmax()]
                    if post_release["visitor_count"].notna().any() else None)
        if peak_row is not None:
            end_date = peak_row["date"]
            if case_files.get("visits_file_daily"):
                end_date_note = "방문자수 피크일 (팀원 제공 진짜 일별 데이터 기준 — 일 단위 정밀도 있음)"
            else:
                end_date_note = "방문자수 피크일 (⚠️ 데이터랩 월값을 일별로 반복 채운 것이라 '일' 단위 정밀도 없음)"
    report["종료점"] = end_date
    report["종료점_기준"] = end_date_note if end_date is not None else "없음(공식대응일/방문자수 모두 미확보)"

    if trigger_date is not None and end_date is not None:
        report["골든타임_갭(일)"] = (pd.Timestamp(end_date) - pd.Timestamp(trigger_date)).days
    else:
        report["골든타임_갭(일)"] = None
        report["갭_계산불가_사유"] = (
            "트리거일 없음" if trigger_date is None else "종료점 데이터 없음(방문자수/공식대응일 필요)"
        )

    return report


def print_report(report: dict):
    print(f"\n=== {report['case_id']} ({report['content_title']}) ===")
    for k, v in report.items():
        if k in ("case_id", "content_title"):
            continue
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="골든타임 계산기")
    parser.add_argument("--case", help="특정 case_id만 계산 (안 주면 전체)")
    args = parser.parse_args()

    case_ids = [args.case] if args.case else list(CASES.keys())
    for case_id in case_ids:
        report = calculate_golden_time(case_id)
        print_report(report)


if __name__ == "__main__":
    main()

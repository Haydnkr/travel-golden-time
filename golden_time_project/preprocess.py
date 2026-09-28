"""
서로 다른 출처(박스오피스/검색량/방문자수)의 시계열을 하나의 표로 정렬하고,
통제변수(주말/공휴일 등) 플래그를 붙이는 모듈.
"""

import pandas as pd

try:
    import holidays as holidays_lib
    _KR_HOLIDAYS = holidays_lib.KR()
except ImportError:  # requirements.txt 설치 전에도 나머지 코드는 돌아가게
    _KR_HOLIDAYS = {}

import config


def build_event_aligned_panel(
    case: dict,
    boxoffice_df: pd.DataFrame,
    search_df: pd.DataFrame,
    visits_df: pd.DataFrame,
) -> pd.DataFrame:
    """세 개의 [date, value] 형태 df를 date 기준으로 합치고,
    개봉일 기준 경과일(days_from_release) 컬럼을 추가함.

    반환 컬럼: date, days_from_release, daily_audience, search_index,
              visitor_count
    """
    df = boxoffice_df.merge(search_df, on="date", how="outer")
    df = df.merge(visits_df, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)

    release = pd.Timestamp(case["release_date"])
    df["days_from_release"] = (df["date"] - release).dt.days

    return df


def add_control_variables(df: pd.DataFrame) -> pd.DataFrame:
    """config.CONTROL_VARIABLES에서 켜진 항목만 컬럼으로 추가.

    is_exam_period은 아직 정의가 없어 전부 False로 채움 —
    TODO(팀): 정의되는 대로 아래 함수 내부만 수정하면 됨.
    """
    df = df.copy()

    if config.CONTROL_VARIABLES.get("is_weekend"):
        df["is_weekend"] = df["date"].dt.weekday >= 5

    if config.CONTROL_VARIABLES.get("is_holiday"):
        df["is_holiday"] = df["date"].apply(lambda d: d.date() in _KR_HOLIDAYS)

    if config.CONTROL_VARIABLES.get("is_exam_period"):
        # TODO(팀): 시험기간 정의(예: 특정 날짜 구간 리스트)가 정해지면 여기 반영
        df["is_exam_period"] = False

    return df


def compute_baseline_excess(df: pd.DataFrame, value_col: str,
                             baseline_days: int = None) -> pd.DataFrame:
    """개봉 전 baseline_days 동안의 평균을 '평시'로 보고,
    이후 값에서 평시 평균을 뺀 초과분(excess) 컬럼을 추가.

    통제변수(주말 등)를 쓰고 싶으면, 이 함수 호출 전에 요일별로 df를 나눠
    각각 호출하는 방식으로 확장 가능 (지금은 단순 전체 평균 버전).
    """
    baseline_days = baseline_days or config.BASELINE_DAYS
    df = df.copy()

    baseline_mask = (df["days_from_release"] < 0) & (
        df["days_from_release"] >= -baseline_days
    )
    baseline_mean = df.loc[baseline_mask, value_col].mean()
    baseline_std = df.loc[baseline_mask, value_col].std()

    df[f"{value_col}_baseline_mean"] = baseline_mean
    df[f"{value_col}_baseline_std"] = baseline_std
    df[f"{value_col}_excess"] = df[value_col] - baseline_mean

    return df

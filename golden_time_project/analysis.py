"""
핵심 분석 로직:
1) 검색량 급증 시점(golden time 시작점 후보) 자동 탐지
2) 골든타임 갭(대응 지연일수) 계산
3) 케이스가 여러 개일 때의 leave-one-case-out 검증 틀
4) 워드클라우드용 키워드 빈도 계산

⚠️ 회의 미결 사항 반영 지점 (config.py의 TODO들과 연결됨):
- GOLDEN_TIME_START_DEF / GOLDEN_TIME_END_DEF 값에 따라 어떤 컬럼을 쓸지 분기
- SPIKE_THRESHOLD_STD_MULTIPLIER 값으로 "급증"의 기준 조정
"""

import re
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

import config


def detect_search_spike_date(df: pd.DataFrame,
                              value_col: str = "search_index_excess",
                              threshold_multiplier: float = None) -> Optional[pd.Timestamp]:
    """평시 대비 (표준편차 * threshold_multiplier)를 넘는 첫 시점을
    '검색량 급증일'로 반환. (compute_baseline_excess 실행 후 호출할 것)

    threshold_multiplier를 안 주면 config.SPIKE_THRESHOLD_STD_MULTIPLIER를 씀.
    sensitivity_analysis()가 이 값을 바꿔가며 여러 번 호출하는 용도로 파라미터화함.

    더 정교한 방법(누적합/CUSUM, 베이지안 change point 등)으로 나중에
    교체하고 싶으면 이 함수만 바꾸면 됨 — 호출부는 그대로 두어도 됨.
    """
    threshold_multiplier = (
        config.SPIKE_THRESHOLD_STD_MULTIPLIER if threshold_multiplier is None
        else threshold_multiplier
    )
    std_col = value_col.replace("_excess", "_baseline_std")
    if std_col not in df.columns:
        raise ValueError(f"{std_col} 없음 — preprocess.compute_baseline_excess 먼저 실행할 것")

    threshold = df[std_col].iloc[0] * threshold_multiplier
    after_release = df[df["days_from_release"] >= 0]
    spikes = after_release[after_release[value_col] > threshold]

    if spikes.empty:
        return None
    return spikes.iloc[0]["date"]


def detect_boxoffice_momentum_date(df: pd.DataFrame,
                                    value_col: str = "daily_audience",
                                    multiplier: float = 1.5) -> Optional[pd.Timestamp]:
    """박스오피스(흥행)용 급증일 탐지 — detect_search_spike_date와 다른 기준을 씀.

    검색량은 개봉 전에도 배경 검색량이 있어서 "평시(개봉 전) 평균 대비 초과"가
    성립하지만, 관객수는 개봉 전엔 정의상 0이라 "평시"라는 게 없음. 그래서
    기준을 "평시 평균"이 아니라 "개봉일(D+0) 관객수"로 바꿔서, 그 값의
    multiplier배를 처음 넘는 날을 찾음.

    ⚠️ 조기경보 취지에 맞춘 설계 결정: "정점(peak)"이 아니라 "개봉일보다
    관객수가 느는 추세로 돌아서는 날"을 봄. 정점은 지나야만 알 수 있는
    값이라 경보 시점으로 못 씀 — 반면 "개봉일보다 관객이 늘고 있다"는
    보통 영화라면 안 보이는 이례적 패턴이라, 그 시점 자체가 "입소문이
    붙기 시작했다"는 조기 신호로 쓸 수 있음.
    """
    after_release = df[df["days_from_release"] >= 0].sort_values("days_from_release")
    if after_release.empty or value_col not in after_release.columns:
        return None

    release_row = after_release[after_release["days_from_release"] == 0]
    if release_row.empty or pd.isna(release_row.iloc[0][value_col]):
        return None
    release_value = release_row.iloc[0][value_col]
    if release_value <= 0:
        return None

    threshold = release_value * multiplier
    after_day0 = after_release[after_release["days_from_release"] > 0]
    momentum = after_day0[after_day0[value_col] > threshold]

    if momentum.empty:
        return None
    return momentum.iloc[0]["date"]


def sensitivity_analysis(
    case: dict,
    aligned_df: pd.DataFrame,
    value_col: str = "search_index",
    baseline_days_grid=(14, 30, 60),
    threshold_grid=(1.5, 2.0, 2.5),
) -> pd.DataFrame:
    """"급증 판정 기준을 조금씩 바꿔도 골든타임 갭이 안정적인가"를 격자탐색으로 확인.

    사례가 1개뿐이라 leave_one_case_out_check로는 검증이 안 될 때, 대신
    쓸 수 있는 최소한의 강건성(robustness) 점검. baseline_days_grid ×
    threshold_grid 조합 각각에 대해 급증일과 골든타임 갭을 계산해서 표로 반환함
    — 결과값(gap_days)이 조합에 따라 크게 안 흔들리면 "이 기준값에 결론이
    자의적으로 좌우되지 않는다"는 근거로 쓸 수 있음.

    aligned_df는 preprocess.build_event_aligned_panel()을 거친 (아직
    compute_baseline_excess는 적용 안 한) 원본 패널을 넣을 것.
    """
    import preprocess  # 함수 내부 import: 최상단에서 돌면 순환참조 위험 있어 지연 로딩

    rows = []
    for baseline_days in baseline_days_grid:
        df_b = preprocess.compute_baseline_excess(aligned_df, value_col, baseline_days=baseline_days)
        for threshold in threshold_grid:
            spike_date = detect_search_spike_date(
                df_b, value_col=f"{value_col}_excess", threshold_multiplier=threshold
            )
            gap_result = compute_golden_time_gap(case, spike_date)
            rows.append({
                "baseline_days": baseline_days,
                "threshold_multiplier": threshold,
                "spike_date": gap_result["spike_date"],
                "days_release_to_spike": gap_result["days_release_to_spike"],
                "gap_days_spike_to_response": gap_result["gap_days_spike_to_response"],
            })

    result = pd.DataFrame(rows)
    if not result["gap_days_spike_to_response"].dropna().empty:
        valid = result["gap_days_spike_to_response"].dropna()
        print(
            f"[민감도 분석] gap_days 범위: {valid.min()}~{valid.max()}일, "
            f"평균 {valid.mean():.1f}일 (조합 {len(result)}개 중 {len(valid)}개 계산됨)"
        )
    return result


def compute_golden_time_gap(case: dict, spike_date, response_date=None) -> dict:
    """골든타임 갭(급증 시점 ~ 공식 대응 시점, 일 단위) 계산.

    response_date를 안 주면 case['official_response_date']를 씀.
    둘 중 하나라도 없으면 gap_days=None으로 반환 (에러 대신).
    """
    response_date = response_date or case.get("official_response_date")
    release_date = pd.Timestamp(case["release_date"])

    result = {
        "case_id": case["case_id"],
        "release_date": release_date,
        "spike_date": pd.Timestamp(spike_date) if spike_date is not None else None,
        "response_date": pd.Timestamp(response_date) if response_date is not None else None,
    }

    if result["spike_date"] is not None:
        result["days_release_to_spike"] = (result["spike_date"] - release_date).days
    else:
        result["days_release_to_spike"] = None

    if result["spike_date"] is not None and result["response_date"] is not None:
        result["gap_days_spike_to_response"] = (
            result["response_date"] - result["spike_date"]
        ).days
    else:
        result["gap_days_spike_to_response"] = None

    return result


def leave_one_case_out_check(cases_results: list) -> pd.DataFrame:
    """케이스가 여러 개 쌓였을 때, gap_days 평균/분산이 얼마나 안정적인지
    보는 아주 단순한 교차검증 틀.

    회의에서 지적된 "결과를 이미 아는 상태로 검증한다"는 문제를,
    최소 3개 이상 케이스가 모이면 "N-1개로 평균 골든타임을 내고
    나머지 1개가 그 범위 안에 드는지" 식으로 확인하는 용도.
    지금은 케이스 1개뿐이라 결과가 비어 나오는 게 정상 — 사례 추가되면 그때 채워짐.
    """
    # gap이 있는 케이스만 남김 (case_id로 짝을 맞춰야 아래 leave-one-out에서
    # 인덱스가 어긋나지 않음 — None 섞여 있을 때 리스트 인덱스만으로 맞추면 틀어짐)
    valid = [(r["case_id"], r["gap_days_spike_to_response"]) for r in cases_results
             if r["gap_days_spike_to_response"] is not None]

    if len(valid) < 2:
        return pd.DataFrame({
            "note": ["사례가 2개 미만이라 교차검증 불가 — 사례 추가 확보 필요"],
            "n_cases_with_gap": [len(valid)],
        })

    all_gaps = [gap for _, gap in valid]
    rows = []
    for i, (case_id, gap) in enumerate(valid):
        others = all_gaps[:i] + all_gaps[i + 1:]
        rows.append({
            "case_id": case_id,
            "held_out_gap": gap,
            "other_cases_mean_gap": float(np.mean(others)) if others else None,
        })
    return pd.DataFrame(rows)


def build_keyword_frequency(texts: list, min_length: int = 2) -> Counter:
    """아주 단순한 한글 키워드 빈도 카운터 (형태소 분석기 없이 임시로 사용).

    TODO(팀/나): 정확한 명사 추출이 필요해지면 konlpy(Okt/Mecab) 등으로 교체.
    지금은 2글자 이상 한글 토큰만 잘라서 빈도를 세는 수준의 임시 버전.
    """
    counter = Counter()
    for text in texts:
        tokens = re.findall(r"[가-힣]{%d,}" % min_length, text)
        counter.update(tokens)
    return counter

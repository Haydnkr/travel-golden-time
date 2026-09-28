"""
스모크 테스트: mock 데이터로 전체 파이프라인이 에러 없이 끝까지 도는지만 확인.
회의에서 정의(config.py)가 바뀌어도 이 테스트는 그대로 통과해야 함 —
통과하지 않으면 어딘가 회의 결과를 코드에 반영하다 실수한 것.

실행: pytest (프로젝트 루트에서)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
import collectors
import preprocess
import analysis
from main import run_case


def test_config_has_at_least_one_case():
    assert len(config.CASES) >= 1


def test_collectors_return_expected_columns():
    case = config.CASES[0]
    start = case["release_date"]
    end = case["release_date"]

    box = collectors.get_kobis_daily_boxoffice(case, start, end, use_mock=True)
    assert {"date", "daily_audience"}.issubset(box.columns)

    search = collectors.get_naver_datalab_trend(case["region_keyword"], start, end, use_mock=True)
    assert {"date", "search_index"}.issubset(search.columns)

    visits = collectors.get_datalab_tourism_visits(case["region_keyword"], start, end, use_mock=True)
    assert {"date", "visitor_count"}.issubset(visits.columns)


def test_preprocess_pipeline_runs():
    case = config.CASES[0]
    start = case["release_date"] - pd.Timedelta(days=10)
    end = case["release_date"] + pd.Timedelta(days=10)

    box = collectors.get_kobis_daily_boxoffice(case, start, end, use_mock=True)
    search = collectors.get_naver_datalab_trend(case["region_keyword"], start, end, use_mock=True)
    visits = collectors.get_datalab_tourism_visits(case["region_keyword"], start, end, use_mock=True)

    df = preprocess.build_event_aligned_panel(case, box, search, visits)
    df = preprocess.add_control_variables(df)
    df = preprocess.compute_baseline_excess(df, "search_index")

    assert "search_index_excess" in df.columns
    assert "days_from_release" in df.columns


def test_golden_time_gap_computation_does_not_crash_without_spike():
    case = config.CASES[0]
    result = analysis.compute_golden_time_gap(case, spike_date=None)
    assert result["gap_days_spike_to_response"] is None


def test_full_case_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = config.CASES[0]
    result = run_case(case)

    assert os.path.exists(result["timeseries_png"])
    assert result["golden_time"]["case_id"] == case["case_id"]

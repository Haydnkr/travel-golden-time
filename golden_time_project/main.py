"""
파이프라인 진입점.

지금은 config.USE_MOCK_DATA = True 상태라 실제 API 키 없이도 바로 실행 가능.
실행: python main.py
결과: output/ 폴더에 케이스별 그래프 + 콘솔에 골든타임 갭 요약표 출력

TODO(팀): API 키 발급되면 config.py의 USE_MOCK_DATA = False 로 바꾸고,
collectors.py의 각 함수 NotImplementedError 부분을 실제 호출 코드로 채우면 됨.
그 외 파일(preprocess/analysis/visualize)은 그대로 재사용 가능하도록 설계함.
"""

import os
from datetime import timedelta

import pandas as pd

import config
import collectors
import preprocess
import analysis
import visualize


def run_case(case: dict) -> dict:
    start = case["release_date"] - timedelta(days=config.WINDOW_BEFORE_DAYS)
    end = case["release_date"] + timedelta(days=config.WINDOW_AFTER_DAYS)

    # 1) 수집
    boxoffice_df = collectors.get_kobis_daily_boxoffice(case, start, end)
    search_df = collectors.get_naver_datalab_trend(case["region_keyword"], start, end)
    search_df = search_df.rename(columns={"search_index": "search_index"})
    visits_df = collectors.get_datalab_tourism_visits(case["region_keyword"], start, end)

    # 2) 전처리
    aligned_df = preprocess.build_event_aligned_panel(case, boxoffice_df, search_df, visits_df)
    aligned_df = preprocess.add_control_variables(aligned_df)
    df = preprocess.compute_baseline_excess(aligned_df, "search_index")

    # 3) 분석
    spike_date = analysis.detect_search_spike_date(df, value_col="search_index_excess")
    golden_time = analysis.compute_golden_time_gap(case, spike_date)
    sensitivity_df = analysis.sensitivity_analysis(case, aligned_df, value_col="search_index")

    # 4) 시각화
    out_png = visualize.plot_event_timeseries(
        df, case, golden_time, out_path=f"output/{case['case_id']}_timeseries.png"
    )

    texts = collectors.get_review_texts(case)
    freq = analysis.build_keyword_frequency(texts)
    try:
        wc_png = visualize.plot_wordcloud(dict(freq), out_path=f"output/{case['case_id']}_wordcloud.png")
    except ImportError:
        wc_png = None  # wordcloud 미설치 시 건너뜀

    return {
        "golden_time": golden_time,
        "timeseries_png": out_png,
        "wordcloud_png": wc_png,
        "sensitivity_df": sensitivity_df,
        "df": df,
    }


def main():
    all_results = []
    for case in config.CASES:
        print(f"\n=== {case['case_id']} ({case['content_title']}) 처리 중 ===")
        result = run_case(case)
        all_results.append(result["golden_time"])
        print(f"  → 그래프 저장: {result['timeseries_png']}")
        if result["wordcloud_png"]:
            print(f"  → 워드클라우드 저장: {result['wordcloud_png']}")

        os.makedirs("output", exist_ok=True)
        sens_path = f"output/{case['case_id']}_sensitivity.csv"
        result["sensitivity_df"].to_csv(sens_path, index=False)
        print(f"  → 민감도 분석 저장: {sens_path}")

    print("\n=== 골든타임 갭 요약 ===")
    summary = pd.DataFrame(all_results)
    print(summary.to_string(index=False))

    print("\n=== 교차검증(사례 2개 이상일 때 유효) ===")
    from analysis import leave_one_case_out_check
    print(leave_one_case_out_check(all_results).to_string(index=False))


if __name__ == "__main__":
    main()

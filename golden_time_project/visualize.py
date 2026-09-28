"""
시각화 모듈. matplotlib(시계열) + wordcloud(키워드).
저장 경로는 output/ 폴더 (main.py에서 os.makedirs로 자동 생성).
"""

import os

import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
    _HAS_WORDCLOUD = True
except ImportError:
    _HAS_WORDCLOUD = False


def plot_event_timeseries(df, case: dict, golden_time_result: dict,
                           out_path: str = "output/event_timeseries.png"):
    """개봉일(t=0) / 검색 급증일 / 공식 대응일을 세로선으로 표시한 시계열 그래프.

    df는 preprocess.build_event_aligned_panel + compute_baseline_excess를
    거친 결과를 기대함 (search_index, visitor_count 컬럼 필요).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(11, 5))

    ax1.plot(df["date"], df["search_index"], color="tab:blue", label="검색량 지수")
    ax1.set_ylabel("검색량 지수", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["visitor_count"], color="tab:orange", label="방문자수")
    ax2.set_ylabel("방문자수", color="tab:orange")

    ax1.axvline(golden_time_result["release_date"], color="gray", linestyle="--",
                label="개봉일")
    if golden_time_result.get("spike_date") is not None:
        ax1.axvline(golden_time_result["spike_date"], color="red", linestyle="--",
                    label="검색 급증일")
    if golden_time_result.get("response_date") is not None:
        ax1.axvline(golden_time_result["response_date"], color="green", linestyle="--",
                    label="공식 대응일")

    plt.title(f"{case['content_title']} × {case['region_keyword']} — 골든타임 타임라인")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_wordcloud(freq_dict: dict, out_path: str = "output/wordcloud.png",
                    font_path: str = None):
    """키워드 빈도 딕셔너리로 워드클라우드 이미지 생성.

    한글 폰트 미지정 시 시스템 기본 폰트라 한글이 네모로 깨질 수 있음 —
    실사용 시 font_path에 나눔고딕 등 한글 폰트(.ttf) 경로를 넣을 것.
    """
    if not _HAS_WORDCLOUD:
        raise ImportError("pip install wordcloud 필요")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    wc = WordCloud(
        font_path=font_path,
        width=800, height=400,
        background_color="white",
    ).generate_from_frequencies(freq_dict)

    wc.to_file(out_path)
    return out_path

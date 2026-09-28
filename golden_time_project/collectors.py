"""
데이터 수집 모듈.

각 함수는 use_mock=True(기본값, config.USE_MOCK_DATA 따라감)일 때 실제 API를
부르지 않고 형태가 같은 가짜(synthetic) 데이터를 반환함 — API 키가 없어도
나머지 파이프라인(전처리/분석/시각화)을 전부 시험해볼 수 있게 하기 위함.

use_mock=False (=.env에 실제 키 설정 후 config.USE_MOCK_DATA=false)이면
실제 API를 호출함. 아래 세 개는 스펙이 고정된 공개 API라 지금 미리 구현해둠:
  - KOBIS 일별 박스오피스
  - 네이버 데이터랩 검색어트렌드
  - 네이버 뉴스 검색
데이터랩(한국관광) 방문자수만 공개 API가 확인되지 않아 CSV 수동 다운로드
경로로 남겨둠 (get_datalab_tourism_visits 참고).
"""

import re
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests

import config

_SESSION = requests.Session()


# ---------------------------------------------------------------------------
# 1) KOBIS 일별 박스오피스
# ---------------------------------------------------------------------------

def get_kobis_daily_boxoffice(case: dict, start, end, use_mock: bool = None) -> pd.DataFrame:
    """영화진흥위원회(KOBIS) 일별 박스오피스에서 case['content_title']에 해당하는
    영화의 일별 관객수를 뽑아옴.

    ⚠️ 알아두어야 할 제약: 이 API는 날짜별 "상위 10위" 영화만 반환함
    (searchDailyBoxOfficeList.json). 흥행작(예: 왕과 사는 남자)은 개봉 후
    한동안 계속 10위 안에 들 가능성이 높아 문제 없지만, 흥행이 약한 대조군
    영화를 나중에 추가하면 특정 날짜에 순위 밖으로 밀려 결측(NaN)이 생길 수
    있음 — 그 경우 daily_audience는 NaN으로 남으니 분석 단계에서 보간(interpolate)
    하거나 결측 그대로 두고 해석에 유의할 것.

    반환: columns = [date, daily_audience]
    """
    use_mock = config.USE_MOCK_DATA if use_mock is None else use_mock
    dates = pd.date_range(start, end, freq="D")

    if use_mock:
        rng = np.random.default_rng(42)
        release = pd.Timestamp(case["release_date"])
        days_from_release = (dates - release).days
        audience = np.where(
            days_from_release < 0,
            0,
            (20000 * np.exp(-((days_from_release - 20) ** 2) / (2 * 30 ** 2))).astype(int),
        )
        audience = audience + rng.integers(0, 500, size=len(dates))
        return pd.DataFrame({"date": dates, "daily_audience": audience})

    config.require_real_key("KOBIS_API_KEY", config.KOBIS_API_KEY)

    rows = []
    title_keyword = case["content_title"]
    for d in dates:
        # KOBIS는 "그 날짜 하루치" 박스오피스를 targetDt로 조회함
        target_dt = d.strftime("%Y%m%d")
        try:
            resp = _SESSION.get(
                "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json",
                params={"key": config.KOBIS_API_KEY, "targetDt": target_dt},
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            movie_list = payload.get("boxOfficeResult", {}).get("dailyBoxOfficeList", [])
        except (requests.RequestException, ValueError):
            movie_list = []

        audience = None
        for movie in movie_list:
            if title_keyword in movie.get("movieNm", ""):
                audience = int(movie.get("audiCnt", 0))
                break

        rows.append({"date": d, "daily_audience": audience})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2) 네이버 데이터랩 검색어트렌드
# ---------------------------------------------------------------------------

def get_naver_datalab_trend(keyword: str, start, end, use_mock: bool = None) -> pd.DataFrame:
    """네이버 데이터랩 검색어트렌드 API.

    ⚠️ 반환값은 조회 구간 내 "최대값을 100으로 한 상대지수"임 — 절대 검색량이
    아니므로, 여러 케이스를 서로 비교(예: 영월 vs 다른 지역)할 때는 절대 수치로
    오해하지 말 것. 같은 케이스 안에서 "개봉 전 대비 개봉 후 몇 배"처럼 상대적으로
    쓰는 용도에는 문제 없음.

    반환: columns = [date, search_index]
    """
    use_mock = config.USE_MOCK_DATA if use_mock is None else use_mock
    dates = pd.date_range(start, end, freq="D")

    if use_mock:
        rng = np.random.default_rng(hash(keyword) % (2 ** 32))
        base = rng.integers(5, 15, size=len(dates)).astype(float)
        spike_at = len(dates) // 3
        base[spike_at:spike_at + 25] += np.linspace(0, 80, 25)
        base[spike_at + 25:] += rng.integers(20, 40, size=len(dates) - spike_at - 25)
        return pd.DataFrame({"date": dates, "search_index": base})

    config.require_real_key("NAVER_CLIENT_ID", config.NAVER_CLIENT_ID)
    config.require_real_key("NAVER_CLIENT_SECRET", config.NAVER_CLIENT_SECRET)

    headers = {
        "X-Naver-Client-Id": config.NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": config.NAVER_CLIENT_SECRET,
        "Content-Type": "application/json",
    }
    body = {
        "startDate": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "endDate": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "timeUnit": "date",
        "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}],
    }

    resp = _SESSION.post(
        "https://openapi.naver.com/v1/datalab/search",
        json=body, headers=headers, timeout=10,
    )
    resp.raise_for_status()
    payload = resp.json()

    data_points = payload["results"][0]["data"]  # [{period: "YYYY-MM-DD", ratio: float}, ...]
    df = pd.DataFrame(data_points).rename(columns={"period": "date", "ratio": "search_index"})
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# 3) 한국관광 데이터랩 방문자수/검색건수 — 공개 API 미확인, CSV 경로로 대체
# ---------------------------------------------------------------------------

def get_datalab_tourism_visits(region_keyword: str, start, end, use_mock: bool = None,
                                csv_path: str = None) -> pd.DataFrame:
    """한국관광 데이터랩 지역별 방문자수 / 관광지 검색건수.

    공개 REST API가 확인되지 않아, 로그인 후 화면에서 엑셀/CSV로 내려받은
    파일을 읽는 방식으로 구현함. csv_path를 안 주면
    `data/raw/datalab_visits_{region_keyword}.csv` 를 찾음.
    해당 CSV는 최소한 date, visitor_count 두 컬럼을 포함해야 함
    (데이터랩 다운로드 파일의 컬럼명이 다르면 이 함수 안에서 rename만 해주면 됨).

    반환: columns = [date, visitor_count]
    """
    use_mock = config.USE_MOCK_DATA if use_mock is None else use_mock
    dates = pd.date_range(start, end, freq="D")

    if use_mock:
        rng = np.random.default_rng(hash(region_keyword) % (2 ** 32))
        base = rng.integers(100, 400, size=len(dates)).astype(float)
        spike_at = len(dates) // 3 + 15  # 검색량 스파이크보다 며칠 늦게 방문 반영(리드타임 흉내)
        base[spike_at:spike_at + 40] += np.linspace(0, 3000, 40)
        base[spike_at + 40:] += rng.integers(1000, 2000, size=len(dates) - spike_at - 40)
        return pd.DataFrame({"date": dates, "visitor_count": base.astype(int)})

    csv_path = csv_path or f"data/raw/datalab_visits_{region_keyword}.csv"
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{csv_path} 가 없습니다. 데이터랩 사이트에서 '{region_keyword}' 방문자수를 "
            f"다운로드해 date, visitor_count 컬럼으로 저장한 뒤 이 경로에 두거나, "
            f"csv_path 인자로 실제 위치를 넘겨주세요."
        ) from e

    missing = {"date", "visitor_count"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} 에 필요한 컬럼이 없습니다: {missing}. "
            f"데이터랩 다운로드 파일의 실제 컬럼명을 이 함수 안에서 rename 해줄 것."
        )
    return df[["date", "visitor_count"]]


# ---------------------------------------------------------------------------
# 4) 네이버 뉴스 검색 (지자체 공식 대응 시점 확인용)
# ---------------------------------------------------------------------------

def get_news_articles(query: str, start, end, use_mock: bool = None,
                       display: int = 100) -> pd.DataFrame:
    """네이버 뉴스 검색 API로 query 관련 기사를 날짜 최신순으로 가져옴.

    자유 텍스트 검색이라 "안전점검"/"위생점검" 등 대응 관련 키워드로
    후보 기사를 뽑은 뒤, 실제 대응일 확정은 사람이 눈으로 보고 판단하는
    반자동 방식을 권장함 (완전 자동화하면 오탐 위험).

    반환: columns = [date, title, url]  (date는 게재일, start~end 범위로
    사후 필터링함 — API 자체에는 날짜 range 파라미터가 없음)
    """
    use_mock = config.USE_MOCK_DATA if use_mock is None else use_mock

    if use_mock:
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-03-01", "2026-03-06"]),
            "title": [f"{query} 관련 위생 점검 착수", f"{query} 안전점검 실시"],
            "url": ["https://example.com/1", "https://example.com/2"],
        })

    config.require_real_key("NAVER_CLIENT_ID", config.NAVER_CLIENT_ID)
    config.require_real_key("NAVER_CLIENT_SECRET", config.NAVER_CLIENT_SECRET)

    headers = {
        "X-Naver-Client-Id": config.NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": config.NAVER_CLIENT_SECRET,
    }

    rows = []
    start_idx = 1
    while start_idx <= 1000:  # 네이버 뉴스 검색 API 최대 조회 한도
        resp = _SESSION.get(
            "https://openapi.naver.com/v1/search/news.json",
            params={"query": query, "display": min(display, 100),
                    "start": start_idx, "sort": "date"},
            headers=headers, timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            break

        for item in items:
            title = re.sub(r"</?b>", "", item["title"])
            pub_date = parsedate_to_datetime(item["pubDate"]).date()
            rows.append({"date": pd.Timestamp(pub_date), "title": title, "link": item["link"]})

        start_idx += len(items)
        if len(items) < 100:
            break

    df = pd.DataFrame(rows).rename(columns={"link": "url"})
    if df.empty:
        return df
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]


# ---------------------------------------------------------------------------
# 5) 워드클라우드용 리뷰/기사 텍스트
# ---------------------------------------------------------------------------

def get_review_texts(case: dict, use_mock: bool = None) -> list:
    """워드클라우드용 리뷰/기사 텍스트 모음.

    실제 연동 시 가장 간단한 방법: 위 get_news_articles()로 뽑은 기사 제목들을
    그대로 texts로 넘기는 것 (본문 크롤링은 저작권·robots.txt 확인 후 진행 권장).
    """
    use_mock = config.USE_MOCK_DATA if use_mock is None else use_mock

    if use_mock:
        return [
            f"{case['content_title']} 보고 {case['region_keyword']} 가고 싶어졌다",
            f"{case['site_keywords'][0]} 실제로 가보니 감동적이었다",
            "가족 단위 관광객이 많이 늘었다는 뉴스를 봤다",
            f"{case['content_title']} 촬영지 투어 다녀왔어요",
        ]

    news_df = get_news_articles(
        f"{case['content_title']} {case['region_keyword']}",
        case["release_date"], pd.Timestamp.today(),
        use_mock=False,
    )
    return news_df["title"].tolist() if not news_df.empty else []

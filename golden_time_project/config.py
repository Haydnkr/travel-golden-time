"""
프로젝트 전역 설정.
2차 회의에서 정의가 바뀌면 이 파일만 고치면 됨 — 다른 코드는 여길 참조만 함.
"""

import os
from datetime import date

try:
    from dotenv import load_dotenv
    load_dotenv()  # 프로젝트 루트의 .env 파일을 읽어 os.environ에 채워줌 (없으면 조용히 무시)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# TODO(회의): 골든타임 시작점 정의
#   "search_spike"  : 검색량이 평시 대비 급증하기 시작한 날 (예린님 제안, 기본값)
#   "release_date"  : 콘텐츠 공개일 그 자체
GOLDEN_TIME_START_DEF = "search_spike"

# TODO(회의): 골든타임 끝점 정의
#   "official_response" : 지자체 공식 대응(안전점검 등) 발표일 (기본값)
#   "visit_peak"         : 실제 방문자수가 정점을 찍은 날
GOLDEN_TIME_END_DEF = "official_response"

# ---------------------------------------------------------------------------
# TODO(회의): 검색량 "급증"의 기준 — 표준편차 배수 방식 (analysis.detect_search_spike_date 참고)
# 평시(개봉 전 BASELINE_DAYS일) 평균 대비 이 배수를 넘는 첫날을 급증일로 봄
SPIKE_THRESHOLD_STD_MULTIPLIER = 2.0
BASELINE_DAYS = 30

# ---------------------------------------------------------------------------
# TODO(회의): 분석 대상 장르 범위 — 우선 "영화"로 한정하기로 함
CONTENT_TYPE_SCOPE = ["movie"]  # 추후 ["movie", "drama"] 등으로 확장 가능

# ---------------------------------------------------------------------------
# 사례(케이스) 목록 — 케이스 추가할 땐 이 리스트에 dict 하나만 추가하면 됨.
# official_response_date 등 실측값이 아직 없는 케이스는 None으로 두면
# 해당 계산은 건너뜀 (에러 나지 않음).
CASES = [
    {
        "case_id": "yeongwol_2026",
        "content_title": "왕과 사는 남자",
        "content_type": "movie",
        "release_date": date(2026, 2, 4),
        "region_keyword": "영월",
        "site_keywords": ["청령포", "장릉", "엄흥도 후손 집성촌"],
        "official_response_date": date(2026, 3, 6),  # 강원도 안전점검 발표(뉴스 기준)
        "notes": (
            "PPSS/경향/문화일보/traveli.net 보도 기준. "
            "※중요: 같은 영화가 '실제 역사지 관광'(영월=단종 유배·서거지)과 "
            "'촬영지 관광'(문경새재 오픈세트=촬영 장소)을 동시에 유발한 이중 사례 — "
            "아래 mungyeong_2026 케이스와 페어로 비교분석 가능"
        ),
    },
    {
        "case_id": "mungyeong_2026",
        "content_title": "왕과 사는 남자",
        "content_type": "movie",
        "release_date": date(2026, 2, 4),
        "region_keyword": "문경새재",
        "site_keywords": ["문경새재 오픈세트", "사정전"],
        "official_response_date": date(2026, 3, 14),  # 문경시 한복체험 프로그램 운영 시작(공식 대응 성격의 첫 조치, 뉴스 기준 재확인 필요)
        "notes": (
            "한스경제/국민일보/스타뉴스/위키트리 보도 기준. yeongwol_2026과 동일 영화, "
            "다른 지역(촬영세트) — 문경시 발표 수치: 2/1~3/22 오픈세트 방문객 37,644명, "
            "전년 동기(23,663명) 대비 +59% (스크린투어리즘 효과로 명시). "
            "official_response_date는 3/14 무료 한복체험 프로그램 운영 시작일로 잠정 기재 — "
            "'안전점검'류 대응이 아니라 '관광 프로그램 확대'류라 영월/예산과는 대응 성격이 다름, "
            "서식4 서술 시 이 차이를 명시할 것. "
            "※ 2026-09-24 결정: 검색량_급증일 = 계산불가(노이즈). golden_time_calculator.py 기본 "
            "임계값(평시 대비 표준편차 2.0배)으로는 '문경' 검색량에서 급증일이 탐지되지 않음 — "
            "'문경'이 일반 지명이라 평소 검색량 자체가 30~100 사이로 들쑥날쑥해서 기준을 못 넘김 "
            "(임계값을 1.5배로 낮추면 2/28, 1.0배면 2/14가 잡히지만, 다른 사례와 다른 기준을 "
            "적용하는 게 되어 기각 — 전 사례 동일 기준 2.0배 유지가 원칙). 축제 등 외부 요인 여부도 "
            "확인함: 문경 주요 축제는 찻사발축제(5월)·오미자축제(9월)·사과축제(10~11월)뿐이고 "
            "1~3월(개봉~공식대응 기간)에는 확인된 축제 없음 — 즉 노이즈의 원인이 축제는 아니며, "
            "'문경'이라는 키워드 자체의 구조적 한계(일반 지명 다의성)로 결론. 따라서 문경은 검색량 "
            "정량 지표 없이 '문경시 공식 발표 수치(37,644명/+59%)'만 정성적 근거로 병기하는 사례로 처리."
        ),
    },
    {
        "case_id": "yesan_2026",
        "content_title": "살목지",
        "content_type": "movie",
        "release_date": date(2026, 4, 8),
        "region_keyword": "예산",
        "site_keywords": ["살목지", "광시면"],
        "official_response_date": date(2026, 4, 14),  # 예산군 SNS 공지, 야간(18~06시) 방문 전면통제
        "notes": (
            "한국일보/뉴시스/충청뉴스/매일신문 보도 기준. 공포영화, 개봉 첫 주말부터 "
            "새벽 시간대 인파(차량 약 100대) 몰림 → 영월 사례(D+30 대응)보다 훨씬 빠른 "
            "D+6 대응 — 급격한 스파이크형 사례로 영월(점진적 증가형)과 대조 가능. "
            "개봉전에도 MBC 심야 공포프로그램에 소개된 적 있는 기존 괴담 명소였다는 점 "
            "주의(기존 인지도 통제변수 관련)"
        ),
    },
    {
        "case_id": "sopoong_2024",
        "content_title": "소풍",
        "content_type": "movie",
        "release_date": date(2024, 2, 7),
        "region_keyword": "남해",
        "site_keywords": ["다랭이마을", "평산마을"],
        "official_response_date": None,  # 팀원 데이터에 방문자/KOBIS는 있으나 공식대응 기록은 아직 미확인
        "notes": (
            "팀원 공유 파일(영화관광_학습용_추가_완성본.xlsx, '확보기간' 시트) 기준 방문자/검색량/KOBIS "
            "데이터 확보됨(신안군 도초면과 동일 형식) — 학습가능여부 '조건부: 통제자료 보완'. "
            "2026-09-27 구글 트렌드 신규 수집: '소풍' 키워드 자체가 '피크닉'이라는 일상적 일반명사라 "
            "노이즈가 있지만(평소에도 20~30대), 개봉일(2/7)부터 뚜렷하게 40~100대로 올라붙는 패턴은 "
            "확인됨(피크 2/10=100) — 일반명사 키워드치고는 사용 가능한 수준."
        ),
    },
    {
        "case_id": "paemyo_2024",
        "content_title": "파묘",
        "content_type": "movie",
        "release_date": date(2024, 2, 22),
        "region_keyword": "무주",
        "site_keywords": ["부남면"],
        "official_response_date": None,  # 미확인 — 1100만 관객 흥행작이라 별도 확인 필요
        "notes": (
            "팀원 공유 파일 기준 방문자/검색량/KOBIS 데이터 확보됨(무주군 부남면) — 학습가능여부 "
            "'조건부: 통제자료 보완'. 실제 촬영지 중 하나로 무주(전북)가 언급됨(파주/고성/춘천/무주/충주 "
            "등 여러 지역 촬영 후 합성). 2026-09-27 구글 트렌드 신규 수집: '파묘'는 흔치 않은 단어라 "
            "개봉전 베이스라인이 거의 0에 수렴 — 지금까지 조사한 사례 중 가장 깨끗한 신호. "
            "개봉일(2/22)부터 급증(43)해서 3/2에 정점(100)."
        ),
    },
    {
        "case_id": "jasaneobo_2021",
        "content_title": "자산어보",
        "content_type": "movie",
        "release_date": date(2021, 3, 31),
        "region_keyword": "신안",
        "site_keywords": ["도초도", "도초면"],
        "official_response_date": None,  # 미확인
        "notes": (
            "팀원 공유 파일 기준 방문자/검색량/KOBIS 데이터 확보됨(신안군 도초면) — 학습가능여부 "
            "'조건부: 통제자료 보완'. 촬영지 신안군 도초면 발매리(도초도) 실존 확인됨. "
            "※2026-09-27 정정: 이전 세션에서 '자산어보는 구글 트렌드 신호가 거의 없다'고 판단해 "
            "사례에서 제외했었는데, 이번에 '자산어보' 키워드로 재수집해보니 개봉전 베이스라인 0 → "
            "개봉일(3/31)=51 → 4/3=73(1차 피크) → 5/1=100(정점, 단 근로자의날 연휴와 겹쳐 해석 "
            "주의) 로 뚜렷한 신호가 확인됨. 이전 결론이 틀렸던 것으로 보이며(다른 키워드로 조회했었을 "
            "가능성), 이 사례는 재검토 대상으로 격상."
        ),
    },
    # TODO(팀): 대조군/추가 사례 확보되는 대로 아래 형식으로 추가
    # {
    #     "case_id": "nonsan_2018",
    #     "content_title": "미스터 션샤인",
    #     "content_type": "drama",
    #     "release_date": date(2018, 7, 7),
    #     "region_keyword": "논산",
    #     "site_keywords": ["선샤인스튜디오"],
    #     "official_response_date": None,
    #     "notes": "방영 종료 8일 전 오픈세트장 개장 — 선제 대응 대조사례",
    # },
    # 리서치 완료, 데이터 확보되는 대로 추가 예정 (공식대응일 미확인 — 확인되면 채울 것)
    # {
    #     "case_id": "gokseong_2016",
    #     "content_title": "곡성",
    #     "content_type": "movie",
    #     "release_date": date(2016, 5, 12),
    #     "region_keyword": "곡성",
    #     "site_keywords": ["섬진강 기차마을"],
    #     "official_response_date": None,
    #     "notes": "제목=지명인 특이 케이스. 개봉 후 휴일마다 30~40팀 방문, 장미축제와 "
    #              "시기 겹침(변수통제 필요). 초기엔 이미지 타격 우려했으나 반전된 사례",
    # },
    # {
    #     "case_id": "busan_2014",
    #     "content_title": "국제시장",
    #     "content_type": "movie",
    #     "release_date": date(2014, 12, 17),  # 정확한 날짜 KOBIS로 재확인 필요
    #     "region_keyword": "국제시장",
    #     "site_keywords": ["꽃분이네"],
    #     "official_response_date": None,
    #     "notes": "1,400만 관객. 관광객 3~4배 증가, 주말 10만명. 촬영지 '꽃분이네'는 "
    #              "임대료 문제로 결국 폐업 — 오버투어리즘 부작용 사례로도 활용 가능",
    # },
    {
        "case_id": "gunwi_2018",
        "content_title": "리틀 포레스트",
        "content_type": "movie",
        "release_date": date(2018, 2, 28),  # 정확한 날짜 KOBIS로 재확인 필요
        "region_keyword": "군위",  # 회의노트엔 "의성"이었으나 핵심 촬영지는 군위군 우보면
        "site_keywords": ["미성리"],
        "official_response_date": None,
        "notes": (
            "주말 100~200명 방문 — 다른 사례 대비 규모가 작은 케이스. "
            "군위군은 2023년 경북→대구광역시 편입, 지역 검색 시 유의. "
            "2026-09-20 팀 대화(예린/이주영) 결론: 2020년 이전이라 관광데이터랩에서 "
            "'지역별 방문자수' 외 나머지 관광 데이터는 수집 불가 확인 — Google Trends도 "
            "'군위'·'리틀포레스트' 키워드, 전국/대구광역시 필터 모두 대부분 결측으로 재확인됨. "
            "→ 팀 합의: 이 사례는 정밀 분석 대상이 아니라 '영화 관람자수 + 지역 방문자수'만으로 "
            "검증하는 test set으로 격하해서 사용"
        ),
    },
]

# ---------------------------------------------------------------------------
# 통제변수 on/off — 회의에서 나온 "연휴/주말/시험" 등을 여기서 켜고 끔
CONTROL_VARIABLES = {
    "is_weekend": True,
    "is_holiday": True,       # 한국 공휴일 (holidays 패키지)
    "is_exam_period": False,  # TODO(팀): 정의되면 preprocess.py에 로직 추가
}

# ---------------------------------------------------------------------------
# 분석 윈도우 — 개봉일 기준 앞뒤 며칠을 볼 것인지
WINDOW_BEFORE_DAYS = 30
WINDOW_AFTER_DAYS = 120

# ---------------------------------------------------------------------------
# API 키 — 실제 값은 이 파일에 직접 쓰지 말고 프로젝트 루트에 .env 파일을 만들어
# (.env.example 참고) KOBIS_API_KEY=실제키 형식으로 넣을 것. .env는 git에 올리지 않음.
KOBIS_API_KEY = os.environ.get("KOBIS_API_KEY", "YOUR_KOBIS_KEY")
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "YOUR_NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "YOUR_NAVER_CLIENT_SECRET")

# .env에 실제 키를 넣으면 자동으로 False처럼 동작해도 되지만, 안전하게 명시적으로 관리.
# 실제 API 키를 넣은 뒤에는 이 값을 False로 바꿀 것. 그전까지는 mock 데이터로
# 파이프라인 전체를 시험 가능.
USE_MOCK_DATA = os.environ.get("USE_MOCK_DATA", "true").lower() != "false"


def _is_placeholder(value: str) -> bool:
    return value.startswith("YOUR_")


def require_real_key(key_name: str, value: str):
    """실제 API를 호출하는 함수 초입에서 호출 — 키가 아직 placeholder면
    바로 알아볼 수 있는 에러를 던짐 (네트워크 에러로 헷갈리지 않게)."""
    if _is_placeholder(value):
        raise RuntimeError(
            f"{key_name}가 아직 설정되지 않았습니다. 프로젝트 루트에 .env 파일을 만들고 "
            f"{key_name}=실제값 을 넣어주세요 (.env.example 참고)."
        )

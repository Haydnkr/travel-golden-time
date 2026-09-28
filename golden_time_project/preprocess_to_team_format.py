"""
data/raw/의 우리 수집 데이터를, 팀원이 공유한
"영화관광_골든타임_분석용_공유본.xlsx"와 같은 시트 구성(일별통합 / 거주지 /
검색량 / 원본목록 / 변수설명)으로 재구성하는 전처리 스크립트.

주의: 팀원 파일의 실제 데이터 값은 전혀 읽어오지 않음 — 시트/컬럼 구성만
참고했고, 채우는 값은 전부 data/raw/의 우리 자체 수집 데이터에서만 가져옴.
우리에게 없는 데이터(예: 관광소비, 읍면별방문, NAVER 검색지수)는 해당 항목을
빈 채로 두거나 시트 자체를 만들지 않음 — 절대 팀원 데이터로 채우지 않음.

실행: python preprocess_to_team_format.py
출력: data/processed/우리팀_데이터_전처리.xlsx
"""

import csv
import datetime
import os

import openpyxl
from openpyxl.styles import Font

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUT_DIR = RAW_DIR
OUT_PATH = os.path.join(OUT_DIR, "우리팀_데이터_전처리.xlsx")

WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]


def read_csv(path, encoding="utf-8-sig"):
    with open(path, encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


def parse_date(s):
    return datetime.datetime.strptime(s.strip(), "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# 사례 정의 — config.py의 CASES와 맞춤 (문경새재는 Google Trends 미수집이라 제외)
CASES = [
    dict(
        지역="영월", 영화="왕과 사는 남자", 개봉일=datetime.date(2026, 2, 4),
        trends_files=["googletrends_yeongwol_daily_partial.csv",
                      "googletrends_yeongwol_daily_part2.csv"],
        trends_keycol="영월", site_keycols=["청령포", "장릉"],
        boxoffice_file="boxoffice_manual_yeongwol_2026.csv",
        boxoffice_format="yeongwol",
        datalab_visits_file="datalab_visits_영월.csv",
        datalab_residence_file="datalab_residence_영월.csv",
    ),
    dict(
        지역="예산", 영화="살목지", 개봉일=datetime.date(2026, 4, 8),
        trends_files=["googletrends_yesan_daily.csv"],
        trends_keycol="예산", site_keycols=["살목지"],
        boxoffice_file="boxoffice_manual_yesan_2026.csv",
        boxoffice_format="yesan",
        datalab_visits_file="datalab_visits_예산.csv",
        datalab_residence_file="datalab_residence_예산.csv",
    ),
    dict(
        지역="곡성", 영화="곡성", 개봉일=datetime.date(2016, 5, 12),
        trends_files=["googletrends_gokseong_daily.csv"],
        trends_keycol="곡성", site_keycols=[],
        boxoffice_file=None, boxoffice_format=None,
        datalab_visits_file=None, datalab_residence_file=None,
    ),
    dict(
        지역="국제시장", 영화="국제시장", 개봉일=datetime.date(2014, 12, 17),
        trends_files=["googletrends_gukjesijang_daily.csv"],
        trends_keycol="국제시장", site_keycols=[],
        boxoffice_file=None, boxoffice_format=None,
        datalab_visits_file=None, datalab_residence_file=None,
    ),
    dict(
        지역="군위", 영화="리틀 포레스트", 개봉일=datetime.date(2018, 2, 28),
        trends_files=["googletrends_gunwi_daily.csv"],
        trends_keycol="군위", site_keycols=[],
        boxoffice_file=None, boxoffice_format=None,
        datalab_visits_file=None, datalab_residence_file=None,
    ),
]


def load_trends(case):
    """여러 파일에 걸친 Google Trends 데이터를 날짜 기준으로 병합(중복 날짜는 뒤 파일 우선).

    초기(partial) 파일은 date,search_index_google_trends,keyword,note 형태의
    긴 포맷(키워드 1개)이라, date,<키워드명>,<키워드명>...,note 형태의 넓은
    포맷으로 정규화해서 병합함."""
    merged = {}
    for fname in case["trends_files"]:
        rows = read_csv(os.path.join(RAW_DIR, fname))
        for row in rows:
            d = row.get("date", "").strip()
            if not d:
                continue
            if "search_index_google_trends" in row:
                keyword = row.get("keyword", "").strip()
                normalized = {"date": d, keyword: row.get("search_index_google_trends", "")}
            else:
                normalized = row
            merged.setdefault(d, {}).update(normalized)
    return merged


def load_boxoffice(case):
    """KOBIS_관객수(정확한 단일일자)와 KOBIS_기타관측(주말합계/누적 등)을 분리."""
    daily, other = {}, {}
    if not case["boxoffice_file"]:
        return daily, other
    rows = read_csv(os.path.join(RAW_DIR, case["boxoffice_file"]))
    fmt = case["boxoffice_format"]
    for row in rows:
        if fmt == "yeongwol":
            d = row["date"].strip()
            daily[d] = row["daily_audience"]
        elif fmt == "yesan":
            key = row["date_or_range"].strip()
            first_date = key.split("~")[0]
            if row["type"] == "daily":
                daily[first_date] = row["audience"]
            else:
                other[first_date] = f"{row['type']}:{row['audience']}({key}, {row['source_note']})"
    return daily, other


def load_datalab_visits(case):
    """월별 방문자수(연인원)를 {기준년월(YYYYMM): 값} 딕셔너리로."""
    if not case["datalab_visits_file"]:
        return {}
    rows = read_csv(os.path.join(RAW_DIR, case["datalab_visits_file"]))
    return {row["기준년월"].strip(): row["방문자수"] for row in rows}


def build_daily_sheet(wb):
    ws = wb.create_sheet("일별통합")
    headers = [
        "날짜", "영화", "지역", "개봉일", "개봉후경과일", "개봉전후", "요일", "주말",
        "GoogleTrends_지역지수", "GoogleTrends_사이트1명", "GoogleTrends_사이트1지수",
        "GoogleTrends_사이트2명", "GoogleTrends_사이트2지수",
        "KOBIS_관객수", "KOBIS_기타관측",
        "관광데이터랩_월방문자수_연인원", "관광데이터랩_해상도", "원본",
    ]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for case in CASES:
        trends = load_trends(case)
        box_daily, box_other = load_boxoffice(case)
        visits_monthly = load_datalab_visits(case)
        for d_str in sorted(trends.keys()):
            row = trends[d_str]
            d = parse_date(d_str)
            days_from_release = (d - case["개봉일"]).days
            site1 = case["site_keycols"][0] if len(case["site_keycols"]) > 0 else ""
            site2 = case["site_keycols"][1] if len(case["site_keycols"]) > 1 else ""
            yyyymm = d.strftime("%Y%m")
            ws.append([
                d, case["영화"], case["지역"], case["개봉일"], days_from_release,
                "개봉전" if days_from_release < 0 else "개봉후", WEEKDAY_KR[d.weekday()],
                1 if d.weekday() >= 5 else 0,
                row.get(case["trends_keycol"]) or None,
                site1 or None, (row.get(site1) if site1 else None),
                site2 or None, (row.get(site2) if site2 else None),
                box_daily.get(d_str), box_other.get(d_str),
                visits_monthly.get(yyyymm),
                ("월값(일별아님, 참고용)" if yyyymm in visits_monthly else None),
                "GoogleTrends" + ("+KOBIS" if d_str in box_daily or d_str in box_other else "")
                + ("+관광데이터랩" if yyyymm in visits_monthly else ""),
            ])
    return ws


def build_residence_sheet(wb):
    ws = wb.create_sheet("거주지")
    headers = ["지역", "영화", "집계기간_시작", "집계기간_종료", "거주지명", "광역지자체_비율(%)", "원본파일", "비고"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    period_by_region = {
        "영월": (datetime.date(2025, 12, 1), datetime.date(2026, 8, 31)),
        "예산": (datetime.date(2026, 2, 1), datetime.date(2026, 8, 31)),
    }
    for case in CASES:
        if not case["datalab_residence_file"]:
            continue
        start, end = period_by_region[case["지역"]]
        rows = read_csv(os.path.join(RAW_DIR, case["datalab_residence_file"]))
        for row in rows:
            ws.append([
                case["지역"], case["영화"], start, end,
                row["거주지명"], row["광역지자체 비율(%)"],
                case["datalab_residence_file"],
                "값이 시군구 단위가 아니라 광역지자체(도/시) 단위 비율이 각 행에 반복 표기됨 — 원본 그대로",
            ])
    return ws


def build_search_source_sheet(wb):
    ws = wb.create_sheet("검색량_원본정보")
    headers = ["지역", "영화", "출처", "수집방식", "값의 의미", "기간"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for case in CASES:
        d_start = min(load_trends(case).keys()) if load_trends(case) else ""
        d_end = max(load_trends(case).keys()) if load_trends(case) else ""
        ws.append([
            case["지역"], case["영화"], "Google Trends(trends.google.com)",
            "공개 웹 UI 네트워크 요청 확인 방식(로그인/API키 불필요, 비공식)",
            "0~100 상대지수(해당 키워드·기간·지역 내 최고치=100)",
            f"{d_start} ~ {d_end}",
        ])
    return ws


def build_source_list_sheet(wb):
    ws = wb.create_sheet("원본목록")
    headers = ["지역", "영화", "파일명", "종류", "기간(추정)", "행수", "출처"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for case in CASES:
        for fname in case["trends_files"]:
            rows = read_csv(os.path.join(RAW_DIR, fname))
            dates = [r["date"] for r in rows if r.get("date")]
            ws.append([case["지역"], case["영화"], fname, "Google Trends 검색량",
                       f"{min(dates)}~{max(dates)}" if dates else "", len(rows), "Google Trends"])
        if case["boxoffice_file"]:
            rows = read_csv(os.path.join(RAW_DIR, case["boxoffice_file"]))
            ws.append([case["지역"], case["영화"], case["boxoffice_file"], "박스오피스(관객수)",
                       "", len(rows), "언론보도(KOBIS 통계 인용)"])
        if case["datalab_visits_file"]:
            rows = read_csv(os.path.join(RAW_DIR, case["datalab_visits_file"]))
            months = [r["기준년월"] for r in rows]
            ws.append([case["지역"], case["영화"], case["datalab_visits_file"], "방문자수(월별)",
                       f"{min(months)}~{max(months)}" if months else "", len(rows), "한국관광 데이터랩"])
        if case["datalab_residence_file"]:
            rows = read_csv(os.path.join(RAW_DIR, case["datalab_residence_file"]))
            ws.append([case["지역"], case["영화"], case["datalab_residence_file"], "방문자 거주지 분포",
                       "", len(rows), "한국관광 데이터랩"])
    return ws


def build_variable_def_sheet(wb):
    ws = wb.create_sheet("변수설명")
    ws.append(["변수", "설명"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    defs = [
        ("날짜/영화/지역", "일별통합 시트의 행 식별자"),
        ("개봉후경과일", "날짜 - 개봉일 (일 단위, 음수=개봉전)"),
        ("GoogleTrends_지역지수", "지역명 키워드의 Google Trends 상대지수(0~100)"),
        ("GoogleTrends_사이트N명/지수", "청령포·장릉·살목지 등 세부 촬영지/명소 키워드와 그 지수"),
        ("KOBIS_관객수", "언론보도 인용 KOBIS 일별 관객수(정확한 단일 날짜인 경우만)"),
        ("KOBIS_기타관측", "주말합계·누적관객수 등 단일일자로 환산 안 되는 관측치(원문 그대로 기록)"),
        ("관광데이터랩_월방문자수_연인원", "한국관광 데이터랩 제공 값 — 월 단위이며, 해당 월의 모든 날짜 행에 동일 값이 반복 표기됨"),
        ("관광데이터랩_해상도", "위 방문자수가 일별이 아니라 월별임을 명시하는 경고 컬럼"),
        ("거주지 시트의 광역지자체_비율(%)", "다운로드 전체 기간 누적 기준 비율. 월별 분해값 아님(우리 원자료 자체가 스냅샷 1건)"),
    ]
    for row in defs:
        ws.append(row)
    return ws


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    build_daily_sheet(wb)
    build_residence_sheet(wb)
    build_search_source_sheet(wb)
    build_source_list_sheet(wb)
    build_variable_def_sheet(wb)

    wb.save(OUT_PATH)
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()

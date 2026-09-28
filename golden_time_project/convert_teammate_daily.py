"""
팀원 공유 파일(영화관광_학습용_추가_완성본.xlsx)의 '일별통합' 시트를
우리 파이프라인(golden_time_calculator.py)이 읽을 수 있는 사례별 CSV로 변환.

팀원 데이터가 우리 것보다 나은 점:
  - 방문자수가 진짜 일별(우리는 데이터랩 월별값을 그 달 모든 날에 반복 채우는
    편법을 썼었음)
  - 박스오피스가 KOBIS(영화진흥위원회) 공식 일별 데이터(우리는 뉴스기사에서
    수동으로 주워모은 몇 개 값뿐이었음)

사용법: python convert_teammate_daily.py
(원본 엑셀 경로가 바뀌면 SOURCE_XLSX만 고치면 됨)
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import openpyxl

SOURCE_XLSX = os.path.join(
    os.path.dirname(__file__), "..", "영화관광_학습용_추가_완성본.xlsx"
)
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

# 영화명(시트 안 표기) -> 우리 쪽에서 쓸 region 슬러그
MOVIE_TO_SLUG = {
    "왕과 사는 남자": "yeongwol",  # 영월군 (문경은 이 시트에 없음 - 그대로 둠)
    "살목지": "yesan",
    "호프": "haenam",
    "리틀 포레스트": "gunwi",
    "소풍": "namhae",
    "파묘": "muju",
    "자산어보": "sinan",
}


def main():
    wb = openpyxl.load_workbook(SOURCE_XLSX, data_only=True)
    ws = wb["일별통합"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    idx = {h: i for i, h in enumerate(header)}

    by_movie = {}
    for r in rows[1:]:
        movie = r[idx["영화"]]
        by_movie.setdefault(movie, []).append(r)

    for movie, slug in MOVIE_TO_SLUG.items():
        recs = by_movie.get(movie)
        if not recs:
            print(f"[스킵] '{movie}' 시트에 없음")
            continue
        recs.sort(key=lambda r: r[idx["날짜"]])

        visits_path = os.path.join(RAW_DIR, f"daily_visits_{slug}.csv")
        box_path = os.path.join(RAW_DIR, f"daily_kobis_{slug}.csv")

        with open(visits_path, "w", encoding="utf-8-sig") as fv, \
             open(box_path, "w", encoding="utf-8-sig") as fb:
            fv.write("date,전체방문자,외지인방문자,현지인방문자\n")
            fb.write("date,KOBIS_관객수,KOBIS_누적관객수\n")
            for r in recs:
                d = r[idx["날짜"]].strftime("%Y-%m-%d")
                total_v = r[idx["전체방문자"]]
                outside_v = r[idx["외지인방문자"]]
                local_v = r[idx["현지인방문자"]]
                if total_v is not None or outside_v is not None or local_v is not None:
                    fv.write(f"{d},{total_v or ''},{outside_v or ''},{local_v or ''}\n")

                kobis_daily = r[idx["KOBIS_관객수"]]
                kobis_cum = r[idx["KOBIS_누적관객수"]]
                if kobis_daily is not None or kobis_cum is not None:
                    fb.write(f"{d},{kobis_daily or ''},{kobis_cum or ''}\n")

        n_visit_rows = sum(1 for r in recs if r[idx["전체방문자"]] is not None)
        n_box_rows = sum(1 for r in recs if r[idx["KOBIS_관객수"]] is not None)
        print(f"[완료] {movie} ({slug}): 방문자 {n_visit_rows}일치, "
              f"KOBIS {n_box_rows}일치 -> {visits_path}, {box_path}")


if __name__ == "__main__":
    main()

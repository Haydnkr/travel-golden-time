"""
전이학습(Transfer Learning) 시도 — 사전학습된 시계열 파운데이션 모델(Amazon Chronos)로
검색량 급증일 탐지를 다시 해봄.

기존 analysis.detect_search_spike_date()의 방식: "개봉 전 30일 평균 대비 표준편차
2배 넘으면 급증"이라는, 우리 데이터 3개짜리 사례에서 직접 정한 규칙(rule-based).

이번 방식: 우리 데이터로 전혀 학습시키지 않은, 이미 수백만 개의 다른 시계열로
사전학습된 모델(Chronos-Bolt)에 "개봉 전 검색량 추이"만 보여주고 "이대로 흘러가면
앞으로 며칠간 어떻게 될 것 같아?"를 예측시킴. 실제 관측값이 모델의 예측 구간
(신뢰구간) 밖으로 벗어나는 첫날 = 모델 입장에서 "이례적"이라고 판단한 날.

핵심 차이: 기존 방식은 우리가 임계값(2.0배)을 임의로 정했지만, 이 방식은
모델이 스스로 "정상 범위"를 판단함 — 우리 데이터가 3개뿐이어도, 모델 자체는
방대한 외부 데이터로 이미 학습되어 있어서 가능함(=전이학습의 핵심 아이디어).

사용법:
    pip install chronos-forecasting  (최초 1회, requirements.txt에도 추가해둠)
    python transfer_learning_chronos.py --case yeongwol_2026
"""

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

import golden_time_calculator as gtc

MODEL_NAME = "amazon/chronos-bolt-small"  # CPU에서도 몇 초 안에 도는 소형 모델
PREDICTION_LENGTH = 45  # 개봉일 이후 며칠까지 볼지
QUANTILES = [0.1, 0.5, 0.9]  # 하위10%/중앙값/상위90% — 상위90%를 "정상 범위 상한"으로 씀


def run(case_id: str):
    case = gtc.CASES[case_id]
    df = gtc._load_search_series(case["search_files"])
    df = df.sort_values("date").reset_index(drop=True)
    release_date = pd.Timestamp(case["release_date"])

    context_df = df[df["date"] < release_date]
    future_df = df[df["date"] >= release_date].head(PREDICTION_LENGTH)

    print(f"\n=== {case_id} ({case['content_title']} / {case['region']}) ===")
    print(f"  개봉 전 컨텍스트: {len(context_df)}일 ({context_df['date'].min().date()} ~ {context_df['date'].max().date()})")
    print(f"  검증할 개봉 후 구간: {len(future_df)}일")

    if len(context_df) < 10 or len(future_df) < 5:
        print("  데이터 부족으로 스킵")
        return

    context = torch.tensor(context_df["search_index"].values, dtype=torch.float32)

    print(f"  사전학습 모델 로딩 중... ({MODEL_NAME})")
    pipeline = BaseChronosPipeline.from_pretrained(MODEL_NAME, device_map="cpu", torch_dtype=torch.float32)

    quantiles, mean = pipeline.predict_quantiles(
        inputs=context, prediction_length=PREDICTION_LENGTH, quantile_levels=QUANTILES,
    )
    # quantiles shape: (1, prediction_length, len(QUANTILES))
    q10 = quantiles[0, :, 0].numpy()
    q50 = quantiles[0, :, 1].numpy()
    q90 = quantiles[0, :, 2].numpy()

    n = len(future_df)
    result = future_df.copy()
    result["예측_중앙값"] = q50[:n]
    result["예측_상위90%"] = q90[:n]
    result["실제"] = result["search_index"].values
    result["이례적(상위90%초과)"] = result["실제"] > result["예측_상위90%"]

    print("\n  날짜        실제   예측중앙값  예측상위90%  이례적?")
    for _, r in result.iterrows():
        flag = " <-- 이례적" if r["이례적(상위90%초과)"] else ""
        print(f"  {r['date'].date()}  {r['실제']:6.1f}   {r['예측_중앙값']:8.1f}   {r['예측_상위90%']:8.1f}  {flag}")

    spike_rows = result[result["이례적(상위90%초과)"]]
    if not spike_rows.empty:
        spike_date = spike_rows.iloc[0]["date"].date()
        gap = (spike_rows.iloc[0]["date"] - release_date).days
        print(f"\n  >>> 전이학습 모델이 감지한 첫 이례적 날짜: {spike_date} (개봉 후 D+{gap}) <<<")
    else:
        print("\n  >>> 예측 구간 안에서 '이례적'으로 판단된 날짜 없음 <<<")

    # 기존 규칙 기반 방식과 비교
    baseline_result = gtc.calculate_golden_time(case_id)
    print(f"\n  [비교] 기존 규칙 기반(표준편차 임계값) 검색량_급증일: {baseline_result['검색량_급증일']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="yeongwol_2026")
    args = parser.parse_args()
    run(args.case)


if __name__ == "__main__":
    main()

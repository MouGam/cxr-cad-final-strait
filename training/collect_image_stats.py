"""
이미지 픽셀 통계 수집 스크립트

NIH ChestX-ray14 전체 이미지(112,120장)에 대해 픽셀 통계를 수집한다.
각 이미지의 mean, std, min, max intensity를 계산하여 CSV로 저장.

이 결과는 quality_filter.py와 preprocess.py에서 품질 필터링 기준으로 사용된다.
- mean < 30: 너무 어두운 이미지 (노출 부족, 장비 오류 등)
- mean > 225: 너무 밝은 이미지 (과노출)
- std < 10: 대비 없는 이미지 (빈 이미지, 단색)

멀티프로세싱(최대 8 workers)으로 병렬 처리하여 속도 최적화.
결과: quality_filter_result/all_image_stats.csv
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "nih-dataset" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "quality_filter_result"


def collect_image_paths(raw_dir: Path) -> list[str]:
    """
    raw 디렉토리 내 모든 PNG 이미지 경로를 수집한다.
    NIH 데이터셋은 images/, images_001/images/ ~ images_011/images/ 구조로 분할 저장됨.
    """
    patterns = [str(raw_dir / "images" / "*.png")]
    for i in range(1, 12):
        # images_001/images/*.png 형태 (표준 구조)
        patterns.append(str(raw_dir / f"images_{i:03d}" / "images" / "*.png"))
        # images_001/*.png 형태 (비표준 구조 대비)
        patterns.append(str(raw_dir / f"images_{i:03d}" / "*.png"))
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    return sorted(set(paths))  # 중복 제거 후 정렬


def analyze_image(path: str) -> tuple:
    """
    단일 이미지의 픽셀 통계를 계산한다.
    Grayscale로 로드하여 mean, std, min, max intensity 반환.
    읽기 실패 시 -1.0으로 표시 (이후 필터링에서 read_error로 처리).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (os.path.basename(path), -1.0, -1.0, -1.0, -1.0)
    return (
        os.path.basename(path),
        round(float(np.mean(img)), 2),
        round(float(np.std(img)), 2),
        float(np.min(img)),
        float(np.max(img)),
    )


def main():
    # 전체 이미지 경로 수집
    paths = collect_image_paths(RAW_DIR)
    print(f"Total images: {len(paths)}")

    # 멀티프로세싱으로 병렬 처리 (chunksize=256으로 IPC 오버헤드 최소화)
    workers = min(cpu_count(), 8)
    print(f"Using {workers} workers")

    results = []
    with Pool(workers) as pool:
        for r in tqdm(pool.imap(analyze_image, paths, chunksize=256),
                      total=len(paths), desc="Collecting stats"):
            results.append(r)

    # DataFrame 생성 및 저장
    df = pd.DataFrame(results, columns=["filename", "mean", "std", "min", "max"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "all_image_stats.csv"
    df.to_csv(out_path, index=False)

    # 결과 요약 출력
    print(f"\nSaved: {out_path}")
    print(f"Total: {len(df)}")
    errors = len(df[df["mean"] < 0])
    if errors:
        print(f"Read errors: {errors}")
    print(f"\nMean intensity — min: {df['mean'].min()}, max: {df['mean'].max()}, "
          f"avg: {df['mean'].mean():.2f}")
    print(f"Std intensity  — min: {df['std'].min()}, max: {df['std'].max()}, "
          f"avg: {df['std'].mean():.2f}")


if __name__ == "__main__":
    main()

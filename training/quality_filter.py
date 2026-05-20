"""
이미지 품질 필터링 스크립트

NIH ChestX-ray14 데이터셋에서 학습에 부적합한 이미지를 탐지하고 분류한다.
의료 이미지 품질 관리(QC) 단계로, 장비 오류/노출 문제/빈 이미지를 제거하여
모델 학습 데이터의 품질을 보장한다.

필터링 기준:
  - too_dark: mean intensity < 30 (노출 부족, 촬영 실패)
  - too_bright: mean intensity > 225 (과노출, 백색 이미지)
  - low_contrast: std < 10 (거의 단색, 빈 이미지)
  - read_error: OpenCV로 읽기 실패 (파일 손상)

결과물:
  - all_image_stats.csv: 전체 이미지 통계 (mean, std, min, max, status)
  - filtered_images.csv: 필터링된 이미지 목록만 추출
  - samples/: 각 카테고리별 샘플 이미지 (최대 20장씩, 수동 검증용)
  - intensity_distribution.png: mean/std 분포 히스토그램 (임계값 시각화)
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서도 실행 가능하도록 백엔드 설정
import matplotlib.pyplot as plt

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "nih-dataset" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "quality_filter_result"

# 필터링 임계값 (의료 이미지 QC 표준 기반)
DARK_THRESHOLD = 30      # mean intensity < 30 → 너무 어두움
BRIGHT_THRESHOLD = 225   # mean intensity > 225 → 너무 밝음
LOW_STD_THRESHOLD = 10   # std < 10 → 대비 없음 (거의 단색)


def collect_image_paths(raw_dir: Path) -> list[str]:
    """
    raw 디렉토리 내 모든 PNG 이미지 경로를 수집한다.
    NIH 데이터셋 구조: images/, images_001/ ~ images_011/
    """
    patterns = [
        str(raw_dir / "images" / "*.png"),
    ]
    for i in range(1, 12):
        patterns.append(str(raw_dir / f"images_{i:03d}" / "*.png"))

    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    return sorted(paths)


def analyze_image(path: str) -> dict:
    """
    단일 이미지의 품질을 분석한다.
    Grayscale로 로드하여 mean/std/min/max intensity를 계산하고,
    임계값 기반으로 품질 상태(status)를 판정한다.

    Returns: dict with path, filename, mean, std, min, max, status
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"path": path, "filename": os.path.basename(path),
                "mean": -1, "std": -1, "min": -1, "max": -1, "status": "read_error"}

    mean_val = float(np.mean(img))
    std_val = float(np.std(img))
    min_val = float(np.min(img))
    max_val = float(np.max(img))

    # 임계값 기반 품질 판정 (우선순위: dark > bright > low_contrast)
    if mean_val < DARK_THRESHOLD:
        status = "too_dark"
    elif mean_val > BRIGHT_THRESHOLD:
        status = "too_bright"
    elif std_val < LOW_STD_THRESHOLD:
        status = "low_contrast"
    else:
        status = "ok"

    return {
        "path": path,
        "filename": os.path.basename(path),
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "min": min_val,
        "max": max_val,
        "status": status,
    }


def save_sample_images(df_filtered: pd.DataFrame, output_dir: Path, max_per_category: int = 20):
    """
    필터링된 이미지를 카테고리별로 샘플 복사한다.
    수동 검증용: 실제로 필터링 기준이 적절한지 눈으로 확인하기 위함.
    각 카테고리(too_dark, too_bright, low_contrast, read_error)별 최대 20장.
    """
    for status in ["too_dark", "too_bright", "low_contrast", "read_error"]:
        subset = df_filtered[df_filtered["status"] == status]
        if len(subset) == 0:
            continue

        cat_dir = output_dir / status
        cat_dir.mkdir(parents=True, exist_ok=True)

        sample = subset.head(max_per_category)
        for _, row in sample.iterrows():
            src = row["path"]
            dst = cat_dir / row["filename"]
            shutil.copy2(src, dst)


def plot_distribution(df: pd.DataFrame, output_dir: Path):
    """
    전체 이미지의 mean intensity와 std intensity 분포를 히스토그램으로 시각화한다.
    임계값(dark/bright/low_contrast)을 점선으로 표시하여 필터링 범위를 시각적으로 확인.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 좌: Mean intensity 분포 + dark/bright 임계값
    axes[0].hist(df["mean"], bins=100, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(DARK_THRESHOLD, color="blue", linestyle="--", label=f"Dark < {DARK_THRESHOLD}")
    axes[0].axvline(BRIGHT_THRESHOLD, color="red", linestyle="--", label=f"Bright > {BRIGHT_THRESHOLD}")
    axes[0].set_xlabel("Mean Pixel Intensity")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Mean Intensity Distribution")
    axes[0].legend()

    # 우: Std intensity 분포 + low_contrast 임계값
    axes[1].hist(df["std"], bins=100, color="coral", edgecolor="black", alpha=0.7)
    axes[1].axvline(LOW_STD_THRESHOLD, color="blue", linestyle="--", label=f"Low Contrast < {LOW_STD_THRESHOLD}")
    axes[1].set_xlabel("Std of Pixel Intensity")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Intensity Std Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "intensity_distribution.png", dpi=150)
    plt.close()


def main():
    print(f"=== Image Quality Filter ===")
    print(f"Thresholds: dark<{DARK_THRESHOLD}, bright>{BRIGHT_THRESHOLD}, low_std<{LOW_STD_THRESHOLD}")
    print()

    # 이미지 경로 수집
    paths = collect_image_paths(RAW_DIR)
    print(f"Total images found: {len(paths)}")

    # 전체 이미지 분석 (순차 처리 — quality_filter는 단일 프로세스로 충분)
    results = []
    for p in tqdm(paths, desc="Analyzing images"):
        results.append(analyze_image(p))

    df = pd.DataFrame(results)

    # 결과 디렉토리 생성 및 요약 출력
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    status_counts = df["status"].value_counts()
    print("\n=== Results ===")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    print(f"  Total filtered: {len(df[df['status'] != 'ok'])}")

    # CSV 저장: 전체 통계 + 필터링 대상만 별도 저장
    df.to_csv(OUTPUT_DIR / "all_image_stats.csv", index=False)
    df_filtered = df[df["status"] != "ok"]
    df_filtered.to_csv(OUTPUT_DIR / "filtered_images.csv", index=False)

    # 샘플 이미지 복사 (수동 검증용)
    save_sample_images(df_filtered, OUTPUT_DIR / "samples")

    # 분포 히스토그램 생성
    plot_distribution(df, OUTPUT_DIR)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - all_image_stats.csv: 전체 이미지 통계")
    print(f"  - filtered_images.csv: 필터링 대상 이미지 목록")
    print(f"  - samples/: 카테고리별 샘플 이미지 (최대 20장씩)")
    print(f"  - intensity_distribution.png: 분포 히스토그램")


if __name__ == "__main__":
    main()

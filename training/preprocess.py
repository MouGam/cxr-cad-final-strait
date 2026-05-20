"""
NIH ChestX-ray14 데이터 전처리 파이프라인

원본 이미지(112,120장)를 학습 가능한 형태로 변환하는 전체 파이프라인.
CLAHE(대비 제한 적응형 히스토그램 균등화)를 적용하여 폐 병변의 시인성을 개선하고,
품질 기준 미달 이미지를 분리한다.

처리 단계:
  Step 1: 수동 선별된 31장(극단적 밝기/저대비)에 CLAHE 적용
  Step 2: mean intensity 기반으로 사전 필터링 목록 생성
  Step 3: 전체 이미지에 CLAHE + Resize(224x224) + 3채널 변환 (멀티프로세싱)
  Step 5: available/unavailable 디렉토리 구성 + 심볼릭 링크
  Step 6: 14개 질환에 대한 Multi-hot Encoding → data.csv 생성
  Step 7: 검증 (파일 수, shape, 라벨 정합성 확인)

입력: nih-dataset/raw/ (원본 이미지 + Data_Entry_2017.csv)
출력: nih-dataset/processed/available/ (111,979장) + unavailable/ (141장)
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
import shutil

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "nih-dataset" / "raw"              # 원본 이미지
BY_HAND_FINAL = PROJECT_ROOT / "by_hand" / "final"          # 수동 처리 이미지 (원본)
BY_HAND_CLAHE = PROJECT_ROOT / "by_hand" / "clahe"          # 수동 처리 이미지 (CLAHE 적용 후)
RESIZED_DIR = PROJECT_ROOT / "nih-dataset" / "resized" / "images"  # CLAHE + Resize 결과
PROCESSED_DIR = PROJECT_ROOT / "nih-dataset" / "processed"  # 최종 출력 디렉토리
STATS_CSV = PROJECT_ROOT / "quality_filter_result" / "all_image_stats.csv"  # 픽셀 통계 CSV

# CLAHE 파라미터: clipLimit=2.0 (대비 제한), tileGridSize=8x8 (로컬 영역 크기)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ─────────────────────────────────────────────
# 수동 처리 대상 31장
# mean intensity 기준으로 자동 필터링에서는 제거되지만,
# 수동 검토 후 CLAHE 적용하면 사용 가능한 이미지들
# ─────────────────────────────────────────────
MANUAL_DARK = [
    "00003465_002.png", "00003465_006.png", "00009621_004.png", "00009621_005.png",
    "00010805_011.png", "00011553_008.png", "00012654_001.png", "00012742_000.png",
    "00014982_000.png", "00015007_005.png", "00015007_006.png", "00015462_000.png",
    "00015462_001.png", "00016292_003.png", "00018251_004.png", "00018251_008.png",
    "00018251_012.png", "00019534_000.png", "00019895_001.png", "00019967_021.png",
    "00022339_000.png", "00022723_000.png", "00022815_012.png", "00027765_000.png",
    "00028474_000.png", "00030320_006.png", "00030609_019.png", "00030609_020.png",
]
MANUAL_WHITE = ["00005618_000.png", "00006094_000.png"]  # 과노출이지만 CLAHE로 복구 가능
MANUAL_STD = ["00004480_000.png"]                         # 저대비이지만 CLAHE로 복구 가능
MANUAL_ALL = set(MANUAL_DARK + MANUAL_WHITE + MANUAL_STD)

# 14개 질환 라벨 (알파벳 순 — 학습 코드와 동일 순서 유지 필수)
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


def build_lookup() -> dict[str, str]:
    """
    raw 디렉토리 내 파일명 → 전체 경로 lookup 딕셔너리를 생성한다.
    NIH 데이터셋이 여러 폴더(images, images_001~011)에 분산되어 있으므로,
    파일명으로 빠르게 경로를 찾기 위한 인덱스.
    """
    lookup = {}
    for p in glob.glob(str(RAW_DIR / "images" / "*.png")):
        lookup[os.path.basename(p)] = p
    for i in range(1, 12):
        for p in glob.glob(str(RAW_DIR / f"images_{i:03d}" / "images" / "*.png")):
            lookup[os.path.basename(p)] = p
    return lookup


def step1_clahe_manual():
    """
    Step 1: 수동 선별된 31장에 CLAHE 적용
    by_hand/final/ → CLAHE → by_hand/clahe/
    이 이미지들은 자동 필터링 기준(mean < 50 등)에 걸리지만,
    수동 검토 결과 CLAHE 적용 시 사용 가능한 것으로 판단됨.
    """
    print("=== Step 1: 31장 CLAHE 적용 ===")
    BY_HAND_CLAHE.mkdir(parents=True, exist_ok=True)

    for f in sorted(os.listdir(BY_HAND_FINAL)):
        if not f.endswith(".png"):
            continue
        img = cv2.imread(str(BY_HAND_FINAL / f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  WARNING: {f} 읽기 실패")
            continue
        result = CLAHE.apply(img)
        cv2.imwrite(str(BY_HAND_CLAHE / f), result)

    print(f"  완료: {len(os.listdir(BY_HAND_CLAHE))}장")


def step2_build_lists(stats: pd.DataFrame) -> tuple[set[str], set[str]]:
    """
    Step 2: 사전 필터링으로 available/unavailable 후보 목록을 생성한다.
    - mean < 50 (수동 처리 28장 제외): 너무 어두움 → 제거
    - mean > 195 (수동 처리 2장 제외): 너무 밝음 → 제거
    CLAHE 후 std 체크는 Step 3에서 동적으로 처리.
    """
    print("\n=== Step 2: 유효/제거 이미지 목록 생성 ===")

    pre_remove = set()

    # 어두운 이미지 필터링 (수동 처리 대상 제외)
    dark = stats[stats["mean"] < 50]
    for _, row in dark.iterrows():
        if row["filename"] not in MANUAL_ALL:
            pre_remove.add(row["filename"])

    # 밝은 이미지 필터링 (수동 처리 대상 제외)
    bright = stats[stats["mean"] > 195]
    for _, row in bright.iterrows():
        if row["filename"] not in MANUAL_ALL:
            pre_remove.add(row["filename"])

    print(f"  사전 제거 (mean 기준): {len(pre_remove)}장")

    all_files = set(stats["filename"].tolist())
    pre_available = all_files - pre_remove

    print(f"  사전 available 후보: {len(pre_available)}장")
    return pre_available, pre_remove


def process_single_image(args):
    """
    단일 이미지 전처리 파이프라인 (멀티프로세싱 worker 함수).
    1. CLAHE 적용 (수동 이미지는 이미 적용된 것을 로드)
    2. CLAHE 후 std 체크 (< 25이면 저대비로 판정하여 제거)
    3. Resize 224x224 (Bilinear interpolation)
    4. Grayscale → 3채널 RGB 변환 (pretrained 모델 입력 요구사항)
    5. PNG로 저장
    """
    filename, src_path, is_manual = args

    if is_manual:
        # 수동 이미지: by_hand/clahe/에서 로드 (Step 1에서 이미 CLAHE 적용됨)
        img = cv2.imread(str(BY_HAND_CLAHE / filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (filename, "read_error")
    else:
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (filename, "read_error")
        # CLAHE 적용: 로컬 대비를 균등화하여 폐 병변(Nodule 등)의 시인성 개선
        img = CLAHE.apply(img)

    # CLAHE 적용 후에도 대비가 낮은 이미지는 제거 (수동 처리 이미지는 제외)
    if filename not in MANUAL_ALL:
        img_std = float(np.std(img))
        if img_std < 25:
            return (filename, "low_std")

    # Resize 224x224: ImageNet pretrained 모델의 표준 입력 크기
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Grayscale → 3채널 복제: pretrained 모델이 3채널 RGB 입력을 요구
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 저장
    cv2.imwrite(str(RESIZED_DIR / filename), img_rgb)
    return (filename, "ok")


def step3_clahe_resize(pre_available: set[str], lookup: dict[str, str]) -> tuple[set[str], set[str]]:
    """
    Step 3: 전체 이미지에 CLAHE + Resize + 3채널 변환 수행.
    멀티프로세싱(최대 8 workers)으로 병렬 처리.
    CLAHE 후 std < 25인 이미지는 추가 제거.
    """
    print("\n=== Step 3: CLAHE + Resize 224x224 + 3채널 ===")
    RESIZED_DIR.mkdir(parents=True, exist_ok=True)

    # 멀티프로세싱 작업 목록 생성
    tasks = []
    for filename in sorted(pre_available):
        is_manual = filename in MANUAL_ALL
        src_path = lookup.get(filename, "")
        tasks.append((filename, src_path, is_manual))

    workers = min(cpu_count(), 8)
    print(f"  {len(tasks)}장 처리 시작 ({workers} workers)")

    post_remove = set()
    ok_count = 0

    with Pool(workers) as pool:
        for result in tqdm(pool.imap(process_single_image, tasks, chunksize=256),
                           total=len(tasks), desc="  Processing"):
            filename, status = result
            if status == "ok":
                ok_count += 1
            else:
                post_remove.add(filename)

    final_available = pre_available - post_remove
    print(f"  CLAHE 후 추가 제거 (std<25): {len(post_remove)}장")
    print(f"  최종 resized: {ok_count}장")
    return final_available, post_remove


def step5_organize(available: set[str], unavailable: set[str],
                   meta_df: pd.DataFrame, lookup: dict[str, str]):
    """
    Step 5: 최종 디렉토리 구조 구성.
    - available/images/: resized 이미지에 대한 심볼릭 링크 (디스크 절약)
    - unavailable/images/: raw 원본을 복사 (전처리 안 된 상태 보존)
    """
    print("\n=== Step 5: available/unavailable 구성 ===")

    for subset_name, subset_files in [("available", available), ("unavailable", unavailable)]:
        subset_dir = PROCESSED_DIR / subset_name / "images"
        subset_dir.mkdir(parents=True, exist_ok=True)

        if subset_name == "available":
            # available: resized 이미지에 대한 심볼릭 링크 생성
            for f in tqdm(sorted(subset_files), desc=f"  {subset_name} symlink"):
                src = RESIZED_DIR / f
                dst = subset_dir / f
                if dst.exists():
                    dst.unlink()
                if src.exists():
                    os.symlink(src.resolve(), dst)
        else:
            # unavailable: 전처리되지 않은 raw 원본을 복사 (분석/검토용)
            for f in tqdm(sorted(subset_files), desc=f"  {subset_name} copy"):
                src = lookup.get(f)
                if src:
                    shutil.copy2(src, subset_dir / f)

    print(f"  available: {len(available)}장")
    print(f"  unavailable: {len(unavailable)}장")


def step6_multihot_encoding(available: set[str], unavailable: set[str],
                            meta_df: pd.DataFrame):
    """
    Step 6: Multi-hot Encoding 수행.
    'Finding Labels' 컬럼(예: "Atelectasis|Effusion")을 14차원 이진 벡터로 변환.
    'No Finding'은 모든 질환이 0인 벡터로 자동 처리된다.

    결과: available/data.csv, unavailable/data.csv
    """
    print("\n=== Step 6: Multi-hot Encoding ===")

    def encode_labels(finding_labels: str) -> list[int]:
        """Finding Labels 문자열을 14차원 multi-hot 벡터로 변환"""
        labels = [l.strip() for l in str(finding_labels).split("|")]
        encoding = [0] * len(DISEASES)
        for label in labels:
            if label in DISEASES:
                idx = DISEASES.index(label)
                encoding[idx] = 1
        return encoding

    for subset_name, subset_files in [("available", available), ("unavailable", unavailable)]:
        subset_df = meta_df[meta_df["Image Index"].isin(subset_files)].copy()

        # 각 질환에 대해 0/1 컬럼 추가
        encodings = subset_df["Finding Labels"].apply(encode_labels)
        for i, disease in enumerate(DISEASES):
            subset_df[disease] = encodings.apply(lambda x: x[i])

        out_path = PROCESSED_DIR / subset_name / "data.csv"
        subset_df.to_csv(out_path, index=False)
        print(f"  {subset_name}/data.csv: {len(subset_df)} rows")


def step7_verify(available: set[str], unavailable: set[str]):
    """
    Step 7: 전처리 결과 검증.
    - 파일 수 확인 (available + unavailable = 112,120)
    - 이미지 shape 확인 (224, 224, 3)
    - Multi-hot encoding 정합성 확인 (No Finding → 모두 0)
    - 수동 처리 31장이 available에 포함되었는지 확인
    """
    print("\n=== 검증 ===")

    resized_count = len(list(RESIZED_DIR.glob("*.png")))
    avail_img_count = len(list((PROCESSED_DIR / "available" / "images").glob("*.png")))
    unavail_img_count = len(list((PROCESSED_DIR / "unavailable" / "images").glob("*.png")))

    avail_csv = pd.read_csv(PROCESSED_DIR / "available" / "data.csv")
    unavail_csv = pd.read_csv(PROCESSED_DIR / "unavailable" / "data.csv")

    print(f"  resized/images/: {resized_count}장")
    print(f"  available/images/: {avail_img_count}장")
    print(f"  available/data.csv: {len(avail_csv)} rows")
    print(f"  unavailable/images/: {unavail_img_count}장")
    print(f"  unavailable/data.csv: {len(unavail_csv)} rows")
    print(f"  합계: {avail_img_count + unavail_img_count}장 (expected: 112120)")

    # 이미지 shape 검증: 224x224x3 (RGB)
    sample = cv2.imread(str(next(RESIZED_DIR.glob("*.png"))))
    print(f"  sample shape: {sample.shape} (expected: (224, 224, 3))")

    # Multi-hot encoding 검증: 'No Finding'은 14개 질환 모두 0이어야 함
    no_finding = avail_csv[avail_csv["Finding Labels"] == "No Finding"]
    if len(no_finding) > 0:
        row = no_finding.iloc[0]
        encoding = [row[d] for d in DISEASES]
        print(f"  No Finding encoding: {encoding} (expected: all zeros)")

    # 수동 처리 31장 포함 여부 확인
    manual_in_avail = sum(1 for f in MANUAL_ALL if f in available)
    print(f"  수동 31장 중 available: {manual_in_avail}/31")


def main():
    """전처리 파이프라인 메인 실행"""
    print("=" * 60)
    print("NIH ChestX-ray14 전처리 파이프라인")
    print("=" * 60)

    # 데이터 로드: 픽셀 통계 CSV + 메타데이터 CSV + 파일 경로 인덱스
    stats = pd.read_csv(STATS_CSV)
    meta_df = pd.read_csv(RAW_DIR / "Data_Entry_2017.csv")
    lookup = build_lookup()
    print(f"전체 이미지: {len(stats)}장, 메타데이터: {len(meta_df)} rows")

    # Step 1: 수동 31장 CLAHE 적용
    step1_clahe_manual()

    # Step 2: 사전 필터링 목록 생성
    pre_available, pre_remove = step2_build_lists(stats)

    # Step 3: 전체 CLAHE + Resize + 3채널 변환
    final_available, post_remove = step3_clahe_resize(pre_available, lookup)

    # 최종 unavailable = 사전 제거 + CLAHE 후 추가 제거
    all_unavailable = pre_remove | post_remove
    print(f"\n최종: available={len(final_available)}, unavailable={len(all_unavailable)}")

    # Step 5: 디렉토리 구성 (심볼릭 링크 + 복사)
    step5_organize(final_available, all_unavailable, meta_df, lookup)

    # Step 6: Multi-hot Encoding → data.csv 생성
    step6_multihot_encoding(final_available, all_unavailable, meta_df)

    # Step 7: 검증
    step7_verify(final_available, all_unavailable)

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()

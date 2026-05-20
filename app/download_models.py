"""
HuggingFace에서 모델 가중치 다운로드 + ONNX 변환

1. HuggingFace에서 .pth 다운로드
2. PyTorch → ONNX 변환 (.onnx 파일 생성)
3. ONNX Runtime으로 추론 시 20~40% 빠름

실행:
  python -m app.download_models
"""

import shutil
from pathlib import Path

import torch

from app.config import BEST_FOLDS, MODEL_CONFIGS, MODELS_DIR, PROJECT_ROOT
from app.models import build_model


MODEL_ASSETS_DIR = PROJECT_ROOT / "model_assets"
MODEL_METADATA_FILES = [
    "thresholds.json",
    "platt_params.json",
    "screening_thresholds.json",
    "confirmatory_thresholds.json",
]


def copy_model_metadata():
    """서빙에 필요한 threshold/calibration JSON을 model_assets에서 models로 복사한다."""
    print(f"\n{'='*60}")
    print("  모델 메타데이터 복사")
    print(f"{'='*60}")

    for arch, cfg in MODEL_CONFIGS.items():
        local_dir: Path = cfg["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)

        asset_dir = MODEL_ASSETS_DIR / arch
        for filename in MODEL_METADATA_FILES:
            src = asset_dir / filename
            dst = local_dir / filename
            if not src.exists():
                print(f"  [WARN] {src} 없음")
                continue
            shutil.copy2(src, dst)
            print(f"  {arch}/{filename} 복사 완료")


def download_model_weights():
    """각 모델의 best fold 가중치만 HuggingFace에서 다운로드한다."""
    from huggingface_hub import hf_hub_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for arch, cfg in MODEL_CONFIGS.items():
        local_dir: Path = cfg["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)

        hf_repo = cfg["hf_repo"]
        hf_subfolder = cfg["hf_subfolder"]
        best_fold = BEST_FOLDS[arch]

        print(f"\n{'='*60}")
        print(f"  다운로드: {arch} (best fold {best_fold})")
        print(f"  리포: {hf_repo}")
        print(f"{'='*60}")

        local_path = local_dir / f"fold{best_fold}.pth"

        if local_path.exists():
            print(f"  [SKIP] fold{best_fold}.pth 이미 존재")
        else:
            hf_filename = f"{hf_subfolder}/fold_{best_fold}.pth"
            print(f"  다운로드 중: fold{best_fold}.pth ... ", end="", flush=True)

            try:
                downloaded = hf_hub_download(
                    repo_id=hf_repo,
                    filename=hf_filename,
                    local_dir=str(local_dir / "_hf_cache"),
                )
                shutil.copy2(downloaded, local_path)
                print("완료")
            except Exception as e:
                print(f"실패\n  ERROR: {e}")

        # HF 캐시 정리
        hf_cache = local_dir / "_hf_cache"
        if hf_cache.exists():
            shutil.rmtree(hf_cache, ignore_errors=True)

        size_mb = local_path.stat().st_size / 1024 / 1024 if local_path.exists() else 0
        print(f"\n  결과: fold{best_fold}.pth ({size_mb:.1f} MB)")


def export_to_onnx():
    """best fold .pth 파일을 ONNX 형식으로 변환한다."""
    print(f"\n{'='*60}")
    print("  ONNX 변환")
    print(f"{'='*60}")

    for arch, cfg in MODEL_CONFIGS.items():
        local_dir: Path = cfg["local_dir"]
        input_size = cfg["input_size"]
        best_fold = BEST_FOLDS[arch]

        pth_path = local_dir / f"fold{best_fold}.pth"
        onnx_path = local_dir / f"fold{best_fold}.onnx"

        if onnx_path.exists():
            print(f"  [SKIP] {arch}/fold{best_fold}.onnx 이미 존재")
            continue

        if not pth_path.exists():
            print(f"  [SKIP] {arch}/fold{best_fold}.pth 없음")
            continue

        print(f"  변환 중: {arch}/fold{best_fold} ... ", end="", flush=True)

        try:
            model = build_model(arch)
            state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            dummy = torch.randn(1, 3, input_size, input_size)

            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                dynamic_axes={
                    "input": {0: "batch"},
                    "output": {0: "batch"},
                },
            )

            size_mb = onnx_path.stat().st_size / 1024 / 1024
            print(f"완료 ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"실패\n  ERROR: {e}")

    print("\n[ONNX 변환 완료]")


if __name__ == "__main__":
    download_model_weights()
    export_to_onnx()
    copy_model_metadata()
    print("\n[다음 단계]")
    print("  uvicorn app.api:app --host 0.0.0.0 --port 8000")

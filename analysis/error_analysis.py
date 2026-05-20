"""
에러 분석 + Grad-CAM 시각화

1. Youden's J threshold 기준 FP/FN 케이스 선별
2. False Positive 5건 Grad-CAM
3. False Negative 5건 Grad-CAM
4. 폐 영역 이탈 5건
5. Shortcut Learning 분석
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
GRADCAM_DIR = OUTPUT_DIR / "gradcam_errors"

DATA_ROOT = Path(os.environ.get("NIH_DATA_ROOT", PROJECT_ROOT / "data" / "nih"))
PROCESSED_IMG_DIR = DATA_ROOT / "processed/available/images"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GRADCAM_LAYER = "features.denseblock4.denselayer16.conv2"


def build_densenet():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Linear(1024, 14), nn.Sigmoid())
    return model


def load_densenet(fold, device):
    path = MODELS_DIR / "densenet121" / f"fold{fold}.pth"
    model = build_densenet().to(device)
    model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    model.eval()
    return model


def get_layer(model, name):
    layer = model
    for part in name.split("."):
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
    return layer


def generate_gradcam(model, tensor, class_idx, layer_name):
    """단일 이미지에 대한 Grad-CAM 생성. (H, W) float numpy 반환."""
    features = []
    gradients = []

    target_layer = get_layer(model, layer_name)
    def fwd_hook(m, i, o):
        features.append(o.detach())
    def bwd_hook(m, gi, go):
        gradients.append(go[0].detach())

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    tensor = tensor.clone().requires_grad_(True)
    output = model(tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    fh.remove()
    bh.remove()

    if not gradients or not features:
        return np.zeros((224, 224))

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features[0]).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()

    if cam.ndim == 0 or cam.max() == 0:
        return np.zeros((224, 224))

    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))
    return cam


def overlay_gradcam(img_bgr, cam, alpha=0.4):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, alpha, heatmap, 1 - alpha, 0)
    return overlay


def load_image_for_gradcam(filename):
    """이미지 로드 → tensor + bgr"""
    img = Image.open(PROCESSED_IMG_DIR / filename).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(img).unsqueeze(0)

    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return tensor, img_bgr


def find_fp_fn_cases(labels, probs, thresholds, meta):
    """FP/FN 케이스를 질환별로 수집"""
    fp_cases = []  # (img_idx, disease_idx, prob, disease_name)
    fn_cases = []

    for i, name in enumerate(DISEASE_LABELS):
        thresh = thresholds.get(name, 0.5)
        preds_binary = (probs[:, i] >= thresh).astype(int)
        gt = labels[:, i].astype(int)

        # False Positive: pred=1, gt=0
        fp_mask = (preds_binary == 1) & (gt == 0)
        fp_indices = np.where(fp_mask)[0]
        # 확률 높은 순으로 정렬 (가장 확신한 오류)
        fp_sorted = fp_indices[np.argsort(-probs[fp_indices, i])]
        for idx in fp_sorted[:3]:
            fp_cases.append((int(idx), i, float(probs[idx, i]), name))

        # False Negative: pred=0, gt=1
        fn_mask = (preds_binary == 0) & (gt == 1)
        fn_indices = np.where(fn_mask)[0]
        fn_sorted = fn_indices[np.argsort(probs[fn_indices, i])]  # 확률 낮은 순
        for idx in fn_sorted[:3]:
            fn_cases.append((int(idx), i, float(probs[idx, i]), name))

    return fp_cases, fn_cases


def compute_cam_outside_ratio(cam, threshold=0.5):
    """히트맵이 폐 영역 밖에 얼마나 있는지 대략 추정.
    이미지 가장자리 20% 영역에 있는 활성화 비율."""
    h, w = cam.shape
    border = int(min(h, w) * 0.15)
    mask = np.zeros_like(cam, dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True

    active = cam > threshold
    if active.sum() == 0:
        return 0.0
    return float((active & mask).sum()) / float(active.sum())


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    GRADCAM_DIR.mkdir(exist_ok=True)

    labels = np.load(OUTPUT_DIR / "labels.npy")
    meta = pd.read_csv(OUTPUT_DIR / "test_metadata.csv")

    # DenseNet fold0 TTA predictions 사용
    probs = np.load(OUTPUT_DIR / "preds_densenet_fold0_tta.npy")

    # Youden's J thresholds 로드
    with open(OUTPUT_DIR / "youden_thresholds.json") as f:
        youden_data = json.load(f)

    # DenseNet 5-fold의 thresholds 사용
    dn_thresholds = youden_data.get("DenseNet 5-fold", {}).get("thresholds", {})

    print("[Finding FP/FN cases...]")
    fp_cases, fn_cases = find_fp_fn_cases(labels, probs, dn_thresholds, meta)

    print(f"  Total FP candidates: {len(fp_cases)}")
    print(f"  Total FN candidates: {len(fn_cases)}")

    model = load_densenet(0, device)

    # ─── False Positive 5건 ───
    print("\n" + "=" * 60)
    print("False Positive 케이스 (5건)")
    print("=" * 60)

    fp_report = []
    seen_images = set()
    count = 0
    for img_idx, disease_idx, prob, disease_name in fp_cases:
        if count >= 5:
            break
        filename = meta.iloc[img_idx]["Image Index"]
        key = (filename, disease_name)
        if key in seen_images:
            continue
        seen_images.add(key)

        tensor, img_bgr = load_image_for_gradcam(filename)
        tensor = tensor.to(device)
        cam = generate_gradcam(model, tensor, disease_idx, GRADCAM_LAYER)
        overlay = overlay_gradcam(img_bgr, cam)

        save_name = f"fp_{count+1}_{disease_name}_{filename}"
        cv2.imwrite(str(GRADCAM_DIR / save_name), overlay)

        outside_ratio = compute_cam_outside_ratio(cam)
        fp_report.append({
            "rank": count + 1,
            "filename": filename,
            "disease": disease_name,
            "probability": prob,
            "threshold": dn_thresholds.get(disease_name, 0.5),
            "cam_outside_ratio": outside_ratio,
            "gradcam_file": save_name,
        })
        print(f"  FP#{count+1}: {filename} | {disease_name} | prob={prob:.3f} | outside={outside_ratio:.2f}")
        count += 1

    # ─── False Negative 5건 ───
    print("\n" + "=" * 60)
    print("False Negative 케이스 (5건)")
    print("=" * 60)

    fn_report = []
    seen_images = set()
    count = 0
    for img_idx, disease_idx, prob, disease_name in fn_cases:
        if count >= 5:
            break
        filename = meta.iloc[img_idx]["Image Index"]
        key = (filename, disease_name)
        if key in seen_images:
            continue
        seen_images.add(key)

        tensor, img_bgr = load_image_for_gradcam(filename)
        tensor = tensor.to(device)
        cam = generate_gradcam(model, tensor, disease_idx, GRADCAM_LAYER)
        overlay = overlay_gradcam(img_bgr, cam)

        save_name = f"fn_{count+1}_{disease_name}_{filename}"
        cv2.imwrite(str(GRADCAM_DIR / save_name), overlay)

        outside_ratio = compute_cam_outside_ratio(cam)
        fn_report.append({
            "rank": count + 1,
            "filename": filename,
            "disease": disease_name,
            "probability": prob,
            "threshold": dn_thresholds.get(disease_name, 0.5),
            "cam_outside_ratio": outside_ratio,
            "gradcam_file": save_name,
        })
        print(f"  FN#{count+1}: {filename} | {disease_name} | prob={prob:.3f} | outside={outside_ratio:.2f}")
        count += 1

    # ─── 폐 영역 이탈 5건 ───
    print("\n" + "=" * 60)
    print("폐 영역 이탈 케이스 (5건)")
    print("=" * 60)

    # 모든 양성 예측에 대해 CAM outside ratio가 높은 케이스
    outside_cases = []
    for i, name in enumerate(DISEASE_LABELS):
        thresh = dn_thresholds.get(name, 0.5)
        positive_mask = probs[:, i] >= thresh
        for idx in np.where(positive_mask)[0][:50]:  # 질환별 50건만 샘플링
            filename = meta.iloc[idx]["Image Index"]
            tensor, img_bgr = load_image_for_gradcam(filename)
            tensor = tensor.to(device)
            cam = generate_gradcam(model, tensor, i, GRADCAM_LAYER)
            ratio = compute_cam_outside_ratio(cam)
            if ratio > 0.3:  # 30% 이상 가장자리에 활성화
                outside_cases.append((idx, i, float(probs[idx, i]), name, ratio, cam, img_bgr, filename))

    # outside ratio 높은 순 정렬
    outside_cases.sort(key=lambda x: -x[4])

    outside_report = []
    for rank, (idx, di, prob, dname, ratio, cam, img_bgr, filename) in enumerate(outside_cases[:5]):
        overlay = overlay_gradcam(img_bgr, cam)
        save_name = f"outside_{rank+1}_{dname}_{filename}"
        cv2.imwrite(str(GRADCAM_DIR / save_name), overlay)
        outside_report.append({
            "rank": rank + 1,
            "filename": filename,
            "disease": dname,
            "probability": prob,
            "cam_outside_ratio": ratio,
            "gradcam_file": save_name,
        })
        print(f"  Outside#{rank+1}: {filename} | {dname} | prob={prob:.3f} | outside={ratio:.2f}")

    # ─── 결과 저장 ───
    report = {
        "false_positives": fp_report,
        "false_negatives": fn_report,
        "outside_lung": outside_report,
        "thresholds_used": dn_thresholds,
    }

    with open(OUTPUT_DIR / "error_analysis.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[완료] 에러 분석 결과: {OUTPUT_DIR / 'error_analysis.json'}")
    print(f"  Grad-CAM 이미지: {GRADCAM_DIR}")


if __name__ == "__main__":
    main()

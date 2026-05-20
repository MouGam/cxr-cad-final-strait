"""
추론 엔진 (ONNX Runtime + PyTorch Grad-CAM)

최적화 전략:
  - 순수 추론: ONNX Runtime (CPU에서 PyTorch 대비 20~40% 빠름)
  - Grad-CAM: PyTorch 모델 (backward pass 필요, lazy loading)
  - inference_time_ms: 추론만 측정 (500ms 기준)
  - gradcam_time_ms: Grad-CAM만 별도 측정
  - Ensemble 시 CLAHE 1회만 적용 후 두 크기로 resize
"""

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from app.config import (
    BEST_FOLDS,
    DEFAULT_THRESHOLDS,
    DISEASE_LABELS,
    GRADCAM_LAYERS,
    MODEL_CONFIGS,
    MODELS_DIR,
)
from app.models import load_model
from app.preprocessing import (
    decode_and_clahe,
    hflip_numpy,
    to_model_input,
    to_onnx_input,
)


def _load_thresholds(arch: str) -> dict[str, float]:
    threshold_file = MODELS_DIR / arch / "thresholds.json"
    if threshold_file.exists():
        with open(threshold_file, encoding="utf-8") as f:
            loaded = json.load(f)
        return {d: loaded.get(d, DEFAULT_THRESHOLDS[d]) for d in DISEASE_LABELS}
    return dict(DEFAULT_THRESHOLDS)


def _load_json_thresholds(arch: str, filename: str) -> dict[str, float] | None:
    filepath = MODELS_DIR / arch / filename
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_platt_params(arch: str) -> dict[str, dict[str, float]] | None:
    platt_file = MODELS_DIR / arch / "platt_params.json"
    if platt_file.exists():
        with open(platt_file, encoding="utf-8") as f:
            return json.load(f)
    return None


def _apply_platt(probs: np.ndarray, platt_params: dict[str, dict[str, float]]) -> np.ndarray:
    """Per-disease Platt Scaling: sigmoid(a * logit + b)"""
    calibrated = np.zeros_like(probs)
    for i, d in enumerate(DISEASE_LABELS):
        p = np.clip(probs[i], 1e-7, 1 - 1e-7)
        logit = np.log(p / (1 - p))
        a = platt_params[d]["a"]
        b = platt_params[d]["b"]
        calibrated[i] = 1.0 / (1.0 + np.exp(-(a * logit + b)))
    return calibrated


# ─────────────────────────────────────────────
# Grad-CAM (PyTorch, backward 필요)
# ─────────────────────────────────────────────

class GradCAMExtractor:
    """forward 1회 + backward N회로 여러 질환 Grad-CAM 생성."""

    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self._features: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks(layer_name)

    def _register_hooks(self, layer_name: str):
        target = self._get_layer(layer_name)

        def fwd(m, i, o):
            self._features = o.detach()

        def bwd(m, gi, go):
            self._gradients = go[0].detach()

        self._hooks.append(target.register_forward_hook(fwd))
        self._hooks.append(target.register_full_backward_hook(bwd))

    def _get_layer(self, name: str) -> nn.Module:
        layer = self.model
        for part in name.split("."):
            layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
        return layer

    def _tensor_to_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        t = tensor[0].detach().cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = ((t * std + mean) * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _make_overlay(self, img_bgr: np.ndarray, h: int) -> str:
        if self._gradients is None or self._features is None:
            return ""
        w = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self._features).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()
        if cam.ndim == 0:
            return ""
        if cam.max() > 0:
            cam = cam / cam.max()
        heatmap = cv2.applyColorMap((cv2.resize(cam, (h, h)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.4, heatmap, 0.6, 0)
        _, buf = cv2.imencode(".png", overlay)
        return base64.b64encode(buf).decode("utf-8")

    def generate_multi(self, tensor: torch.Tensor, class_indices: list[int]) -> dict[int, str]:
        if not class_indices:
            return {}
        t = tensor.clone().requires_grad_(True)
        output = self.model(t)
        img_bgr = self._tensor_to_bgr(t)
        h = tensor.shape[2]
        results = {}
        for i, ci in enumerate(class_indices):
            self.model.zero_grad()
            output[0, ci].backward(retain_graph=(i < len(class_indices) - 1))
            results[ci] = self._make_overlay(img_bgr, h)
        return results

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────
# 추론 엔진
# ─────────────────────────────────────────────

class InferenceEngine:
    def __init__(self):
        self._onnx_sessions: dict[str, dict[int, ort.InferenceSession]] = {} if HAS_ONNX else {}
        self._torch_models: dict[str, dict[int, nn.Module]] = {}
        self._thresholds: dict[str, dict[str, float]] = {}
        self._platt_params: dict[str, dict[str, dict[str, float]] | None] = {}
        self._device = self._get_device()
        self._models_ready = False
        self._executor = ThreadPoolExecutor(max_workers=2)
        print(f"[InferenceEngine] device={self._device}, ONNX={HAS_ONNX}")

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def warm_up(self):
        """시작 시 best fold ONNX 세션을 미리 로드한다."""
        for arch in MODEL_CONFIGS:
            best_fold = BEST_FOLDS[arch]
            self._get_onnx_session(arch, best_fold)
        self._models_ready = True
        print("[InferenceEngine] warm-up 완료")

    # ─── ONNX Runtime (빠른 추론용) ───

    def _get_onnx_session(self, arch: str, fold: int):
        if arch not in self._onnx_sessions:
            self._onnx_sessions[arch] = {}
        if fold not in self._onnx_sessions[arch]:
            onnx_path = MODELS_DIR / arch / f"fold{fold}.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX 파일 없음: {onnx_path}\n"
                    "python -m app.download_models 실행 후 재시도하세요."
                )
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            self._onnx_sessions[arch][fold] = ort.InferenceSession(
                str(onnx_path), opts, providers=["CPUExecutionProvider"]
            )
            print(f"[InferenceEngine] ONNX 로드: {arch} fold{fold}")
        return self._onnx_sessions[arch][fold]

    def _onnx_forward(self, session, arr: np.ndarray) -> np.ndarray:
        """ONNX Runtime forward. arr: (1,3,H,W) float32 → (14,) probs."""
        out = session.run(None, {"input": arr})
        return out[0][0]  # (14,)

    # ─── PyTorch (Grad-CAM용, lazy loading) ───

    def _get_torch_model(self, arch: str, fold: int) -> nn.Module:
        if arch not in self._torch_models:
            self._torch_models[arch] = {}
        if fold not in self._torch_models[arch]:
            weight_path = MODELS_DIR / arch / f"fold{fold}.pth"
            if not weight_path.exists():
                raise FileNotFoundError(f"가중치 파일 없음: {weight_path}")
            print(f"[InferenceEngine] PyTorch 로드 (Grad-CAM): {arch} fold{fold}")
            self._torch_models[arch][fold] = load_model(arch, str(weight_path), self._device)
        return self._torch_models[arch][fold]

    def _get_thresholds(self, arch: str) -> dict[str, float]:
        if arch not in self._thresholds:
            self._thresholds[arch] = _load_thresholds(arch)
        return self._thresholds[arch]

    def _get_platt_params(self, arch: str) -> dict[str, dict[str, float]] | None:
        if arch not in self._platt_params:
            self._platt_params[arch] = _load_platt_params(arch)
        return self._platt_params[arch]

    # ─── 추론 (ONNX) ───

    def _predict_single(self, arch: str, fold: int, arr: np.ndarray, use_tta: bool) -> np.ndarray:
        session = self._get_onnx_session(arch, fold)
        p1 = self._onnx_forward(session, arr)
        if use_tta:
            p2 = self._onnx_forward(session, hflip_numpy(arr))
            return (p1 + p2) / 2.0
        return p1

    def _apply_threshold(self, probs: np.ndarray, thresholds: dict[str, float]) -> list[str]:
        return [d for i, d in enumerate(DISEASE_LABELS) if probs[i] >= thresholds[d]]

    # ─── Grad-CAM (PyTorch) ───

    def _generate_gradcam(
        self, arch: str, fold: int, clahe_img: np.ndarray, diseases: list[str],
    ) -> dict[str, str]:
        if not diseases:
            return {}
        layer_name = GRADCAM_LAYERS.get(arch, "")
        if not layer_name:
            return {}
        model = self._get_torch_model(arch, fold)
        input_size = MODEL_CONFIGS[arch]["input_size"]
        tensor = to_model_input(clahe_img, input_size).to(self._device)
        extractor = GradCAMExtractor(model, layer_name)
        indices = [DISEASE_LABELS.index(d) for d in diseases]
        idx_to_b64 = extractor.generate_multi(tensor, indices)
        extractor.remove_hooks()
        return {diseases[i]: idx_to_b64.get(indices[i], "") for i in range(len(diseases))}

    # ─── 메인 predict ───

    def predict(
        self,
        image_bytes: bytes,
        model_choice: str = "ensemble",
        fold_choice: str = "best",
        threshold_mode: str = "default",
        threshold_value: float = 0.5,
        use_tta: bool = True,
        generate_gradcam: bool = True,
        gradcam_model: str = "densenet",
        gradcam_top1_only: bool = False,
    ) -> dict:
        start = time.time()
        log: list[dict] = []

        def _log(step: str):
            log.append({"step": step, "elapsed_ms": int((time.time() - start) * 1000)})

        _log("이미지 수신")

        # 전처리: CLAHE 1회
        clahe_img = decode_and_clahe(image_bytes)
        _log("전처리 (CLAHE)")

        # ─── 추론 (ONNX Runtime) ───
        infer_start = time.time()

        probs, cam_arch, cam_fold = self._run_onnx_inference(
            clahe_img, model_choice, fold_choice, use_tta
        )

        inference_time_ms = int((time.time() - infer_start) * 1000)

        model_label = {
            "ensemble": f"Ensemble (DenseNet f{BEST_FOLDS['densenet121']} + EfficientNet f{BEST_FOLDS['efficientnet_b4']})",
            "densenet": "DenseNet-121",
            "efficientnet": "EfficientNet-B4",
        }.get(model_choice, model_choice)
        tta_label = " + TTA(H-Flip)" if use_tta else ""
        _log(f"ONNX 추론 완료 — {model_label}{tta_label} ({inference_time_ms}ms)")

        # Per-disease Platt Scaling 적용 (calibration, ECE < 0.01 달성)
        platt_arch = cam_arch if model_choice != "ensemble" else "densenet121"
        platt = self._get_platt_params(platt_arch)
        if platt is not None:
            probs = _apply_platt(probs, platt)
            _log("Platt Scaling 적용")

        # Ensemble에서 Grad-CAM 모델 선택
        if model_choice == "ensemble":
            if gradcam_model == "efficientnet":
                cam_arch = "efficientnet_b4"
                cam_fold = BEST_FOLDS["efficientnet_b4"]
            else:
                cam_arch = "densenet121"
                cam_fold = BEST_FOLDS["densenet121"]

        # Threshold
        if threshold_mode == "custom":
            thresholds = {d: threshold_value for d in DISEASE_LABELS}
        elif threshold_mode == "fixed":
            thresholds = {d: 0.5 for d in DISEASE_LABELS}
        else:
            thresholds = self._get_thresholds(cam_arch)

        detected = self._apply_threshold(probs, thresholds)
        _log(f"Threshold 적용 — 탐지: {len(detected)}개 질환")

        # ─── Grad-CAM (PyTorch, 별도 측정) ───
        gradcam_result = {}
        gradcam_time_ms = 0
        if generate_gradcam and detected:
            cam_start = time.time()
            if gradcam_top1_only:
                cam_targets = [max(detected, key=lambda d: probs[DISEASE_LABELS.index(d)])]
            else:
                cam_targets = detected
            _log(f"Grad-CAM 시작 — {cam_targets} ({cam_arch})")
            gradcam_result = self._generate_gradcam(cam_arch, cam_fold, clahe_img, cam_targets)
            gradcam_time_ms = int((time.time() - cam_start) * 1000)
            _log(f"Grad-CAM 완료 ({gradcam_time_ms}ms)")

        total_ms = int((time.time() - start) * 1000)
        _log(f"전체 완료 — 추론: {inference_time_ms}ms + Grad-CAM: {gradcam_time_ms}ms = {total_ms}ms")

        # Top-1 질환
        top1_idx = int(np.argmax(probs))
        top1_disease = DISEASE_LABELS[top1_idx]
        top1_probability = float(probs[top1_idx])

        # Operating point thresholds (screening / confirmatory)
        thresh_arch = cam_arch if model_choice != "ensemble" else "densenet121"
        screening_thresholds = _load_json_thresholds(thresh_arch, "screening_thresholds.json") or {}
        confirmatory_thresholds = _load_json_thresholds(thresh_arch, "confirmatory_thresholds.json") or {}

        return {
            "predictions": {d: float(probs[i]) for i, d in enumerate(DISEASE_LABELS)},
            "thresholds": thresholds,
            "screening_thresholds": screening_thresholds,
            "confirmatory_thresholds": confirmatory_thresholds,
            "detected": detected,
            "top1_disease": top1_disease,
            "top1_probability": top1_probability,
            "gradcam_base64": gradcam_result,
            "inference_time_ms": inference_time_ms,
            "gradcam_time_ms": gradcam_time_ms,
            "log": log,
            "config": {
                "model": model_choice,
                "fold": fold_choice,
                "tta": use_tta,
                "threshold_mode": threshold_mode,
                "gradcam_model": cam_arch if model_choice == "ensemble" else model_choice,
            },
        }

    def _run_onnx_inference(
        self, clahe_img: np.ndarray, model_choice: str, fold_choice: str, use_tta: bool,
    ) -> tuple[np.ndarray, str, int]:
        """ONNX Runtime 추론. (probs, cam_arch, cam_fold) 반환."""

        if model_choice == "ensemble":
            dn_fold = BEST_FOLDS["densenet121"]
            eff_fold = BEST_FOLDS["efficientnet_b4"]
            arr_dn = to_onnx_input(clahe_img, MODEL_CONFIGS["densenet121"]["input_size"])
            arr_eff = to_onnx_input(clahe_img, MODEL_CONFIGS["efficientnet_b4"]["input_size"])
            fut_dn = self._executor.submit(self._predict_single, "densenet121", dn_fold, arr_dn, use_tta)
            fut_eff = self._executor.submit(self._predict_single, "efficientnet_b4", eff_fold, arr_eff, use_tta)
            p_dn = fut_dn.result()
            p_eff = fut_eff.result()
            probs = (p_dn + p_eff) / 2.0
            return probs, "densenet121", dn_fold

        arch = "densenet121" if model_choice == "densenet" else "efficientnet_b4"
        cfg = MODEL_CONFIGS[arch]
        arr = to_onnx_input(clahe_img, cfg["input_size"])

        if fold_choice == "best":
            fold = BEST_FOLDS[arch]
            probs = self._predict_single(arch, fold, arr, use_tta)
        elif fold_choice == "all":
            preds = [self._predict_single(arch, f, arr, use_tta) for f in range(cfg["num_folds"])]
            probs = np.mean(preds, axis=0)
            fold = BEST_FOLDS[arch]
        else:
            fold = int(fold_choice)
            probs = self._predict_single(arch, fold, arr, use_tta)

        return probs, arch, fold

    @property
    def models_available(self) -> dict[str, list[int]]:
        available = {}
        for arch in MODEL_CONFIGS:
            arch_dir = MODELS_DIR / arch
            folds = [f for f in range(MODEL_CONFIGS[arch]["num_folds"])
                     if (arch_dir / f"fold{f}.onnx").exists()]
            if folds:
                available[arch] = folds
        return available

    @property
    def is_ready(self) -> bool:
        return self._models_ready


engine = InferenceEngine()

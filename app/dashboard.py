"""
Streamlit 판독 보조 대시보드

시연용 단일 페이지:
  - 다중 이미지 업로드 + "분석 시작" 버튼
  - 2단계 호출: 추론 먼저 표시 -> Grad-CAM 이후 추가
  - 좌우 2컬럼 레이아웃 (좌=이미지+Grad-CAM, 우=탐지결과+막대그래프)
  - 처리 로그 상세 표시 (단계별 elapsed_ms)

Streamlit은 모델 파일을 직접 로드하지 않고 반드시 FastAPI와 HTTP 통신만 사용 (MSA 준수).
"""

import base64
import io

import requests
import streamlit as st
from PIL import Image

from app.config import (
    DISEASE_LABELS,
    DISEASE_LABELS_KO,
    STREAMLIT_API_URL,
)

st.set_page_config(
    page_title="CXR-CAD: Chest X-ray AI Detection",
    page_icon="🫁",
    layout="wide",
)

API_URL = STREAMLIT_API_URL


# ─────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────

def check_api_status() -> tuple[bool, str]:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("model_loaded", False), "연결됨"
        return False, f"오류 (HTTP {resp.status_code})"
    except requests.exceptions.ConnectionError:
        return False, "연결 실패"
    except Exception as e:
        return False, f"오류: {e}"


def get_risk_color(prob: float, youden_t: float, screen_t: float | None, confirm_t: float | None) -> str:
    """요구사항 기준 위험도 색상: 빨강 >=0.5, 노랑 0.3~0.5, 초록 <0.3."""
    if prob >= 0.5:
        return "🔴"
    if prob >= 0.3:
        return "🟡"
    return "🟢"


def get_bar_hex(prob: float, youden_t: float, screen_t: float | None, confirm_t: float | None) -> str:
    if prob >= 0.5:
        return "#FF4B4B"
    if prob >= 0.3:
        return "#FFA500"
    return "#21C354"


def call_predict_api(
    image_bytes: bytes,
    filename: str,
    gradcam: bool = True,
    gradcam_top1_only: bool = True,
) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            files={"file": (filename, image_bytes, "image/png")},
            data={
                "model": "ensemble",
                "fold": "best",
                "threshold_mode": "default",
                "tta": "true",
                "gradcam": str(gradcam).lower(),
                "gradcam_model": "densenet",
                "gradcam_top1_only": str(gradcam_top1_only).lower(),
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API 오류 ({resp.status_code}): {resp.json().get('detail', '알 수 없는 오류')}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("FastAPI 서버에 연결할 수 없습니다.")
        return None
    except Exception as e:
        st.error(f"요청 오류: {e}")
        return None


def render_processing_log(log: list[dict]):
    """처리 단계별 로그를 터미널 스타일 텍스트로 표시."""
    if not log:
        return
    with st.expander("처리 로그", expanded=False):
        lines = []
        for entry in log:
            elapsed = entry.get("elapsed_ms", 0)
            step = entry.get("step", "")
            lines.append(f"{elapsed:>6}ms  {step}")
        st.code("\n".join(lines), language=None)


def render_bar_chart(
    predictions: dict,
    thresholds: dict,
    detected: list,
    screening_thresholds: dict | None = None,
    confirmatory_thresholds: dict | None = None,
):
    """14개 질환 수평 막대그래프 (Youden/Screening/Confirmatory threshold 표시)."""
    sorted_diseases = sorted(DISEASE_LABELS, key=lambda d: predictions[d], reverse=True)

    for disease in sorted_diseases:
        prob = predictions[disease]
        threshold = thresholds.get(disease, 0.5)
        screen_t = screening_thresholds.get(disease) if screening_thresholds else None
        confirm_t = confirmatory_thresholds.get(disease) if confirmatory_thresholds else None
        color = get_risk_color(prob, threshold, screen_t, confirm_t)
        bar_hex = get_bar_hex(prob, threshold, screen_t, confirm_t)
        is_detected = disease in detected
        display_name = DISEASE_LABELS_KO.get(disease, disease)
        width_pct = max(prob * 100, 1)
        thresh_pct = min(max(threshold * 100, 0.5), 99.5)

        # Threshold 정보 텍스트
        thresh_info = f"Youden: {threshold:.4f}"
        if screen_t is not None:
            thresh_info += f" | Screen: {screen_t:.4f}"
        if confirm_t is not None:
            thresh_info += f" | Confirm: {confirm_t:.4f}"

        if is_detected:
            label_html = (
                f'<span style="font-weight:700;">{color} {display_name}</span>'
                f'&nbsp; <code>{prob:.3f}</code>'
                f'&nbsp; <small style="color:#888;">{thresh_info}</small>'
            )
        else:
            label_html = (
                f'<span style="color:#999;">{color} {display_name}</span>'
                f'&nbsp; <code style="color:#999;">{prob:.3f}</code>'
                f'&nbsp; <small style="color:#bbb;">{thresh_info}</small>'
            )

        # 막대 + threshold 마커들
        markers_html = ""
        # Youden (검은선)
        markers_html += (
            f'<div style="position:absolute;left:{thresh_pct}%;top:0;width:2px;height:100%;'
            f'background:#333;" title="Youden: {threshold:.4f}"></div>'
        )
        # Screening (파란 점선)
        if screen_t is not None:
            s_pct = min(max(screen_t * 100, 0.5), 99.5)
            markers_html += (
                f'<div style="position:absolute;left:{s_pct}%;top:0;width:2px;height:100%;'
                f'background:#4A90D9;opacity:0.7;" title="Screening: {screen_t:.4f}"></div>'
            )
        # Confirmatory (빨간 점선)
        if confirm_t is not None:
            c_pct = min(max(confirm_t * 100, 0.5), 99.5)
            markers_html += (
                f'<div style="position:absolute;left:{c_pct}%;top:0;width:2px;height:100%;'
                f'background:#D94A4A;opacity:0.7;" title="Confirmatory: {confirm_t:.4f}"></div>'
            )

        bar_html = (
            f'<div style="position:relative;background:#eee;border-radius:4px;height:16px;'
            f'margin:2px 0 6px 0;">'
            f'<div style="background:{bar_hex};width:{width_pct}%;height:100%;border-radius:4px;'
            f'opacity:{"1.0" if is_detected else "0.4"};"></div>'
            f'{markers_html}'
            f'</div>'
        )

        st.markdown(label_html + bar_html, unsafe_allow_html=True)


def render_single_result(image_bytes: bytes, result: dict, gradcam_result: dict | None = None):
    """단일 이미지 결과를 좌우 2컬럼으로 표시."""
    predictions = result["predictions"]
    thresholds = result["thresholds"]
    detected = result["detected"]
    inference_ms = result["inference_time_ms"]
    top1 = result.get("top1_disease", "")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        original_img = Image.open(io.BytesIO(image_bytes))
        st.image(original_img, caption="원본 X-ray", use_container_width=True)

        # Grad-CAM 표시
        gradcam_dict = gradcam_result or result.get("gradcam_base64", {})
        if gradcam_dict:
            for disease_name, b64 in gradcam_dict.items():
                if b64:
                    cam_img = Image.open(io.BytesIO(base64.b64decode(b64)))
                    display = DISEASE_LABELS_KO.get(disease_name, disease_name)
                    st.image(cam_img, caption=f"Grad-CAM: {display}", use_container_width=True)

    with col_right:
        # 탐지 결과 요약
        if detected:
            detected_ko = [DISEASE_LABELS_KO.get(d, d) for d in detected]
            st.warning(f"⚠️ **탐지된 질환 ({len(detected)}개):** {', '.join(detected_ko)}")
        else:
            st.success("✅ 탐지된 질환 없음")

        st.caption(f"추론: {inference_ms}ms | Ensemble (DenseNet f0 + EfficientNet f3) + TTA | Top-1: {top1}")

        # 처리 로그
        render_processing_log(result.get("log", []))

        st.markdown("---")
        render_bar_chart(
            predictions, thresholds, detected,
            screening_thresholds=result.get("screening_thresholds"),
            confirmatory_thresholds=result.get("confirmatory_thresholds"),
        )


# ─────────────────────────────────────────────
# 메인 UI
# ─────────────────────────────────────────────

def main():
    st.title("🫁 CXR-CAD: Chest X-ray AI Detection System")
    st.caption("교육 목적으로 개발된 시스템. 실제 임상 진단에 사용 불가.")

    # 사이드바: API 상태
    with st.sidebar:
        model_loaded, status_msg = check_api_status()
        status_icon = "🟢" if model_loaded else "🔴"
        st.markdown(f"**API 상태:** {status_icon} {status_msg}")
        if not model_loaded:
            st.warning("모델이 로드되지 않았습니다. 서버 초기화를 기다려주세요.")
        st.markdown("---")
        st.markdown(f"**API URL:** `{API_URL}`")
        st.markdown("**설정:** Ensemble (DenseNet f0 + EfficientNet f3) + TTA + Grad-CAM")
        st.markdown("*확률은 Per-disease Platt Scaling 적용 (calibrated)*")
        st.markdown("---")
        st.markdown(
            "**색상 기준 (요구사항 고정)**\n"
            "- 🔴 높음: prob >= 0.5\n"
            "- 🟡 주의: 0.3 <= prob < 0.5\n"
            "- 🟢 낮음: prob < 0.3\n\n"
            "**막대 위 마커**\n"
            "- ■ 검정 = Youden's J (균형점)\n"
            "- ■ 파랑 = 스크리닝 (Sens>=90%)\n"
            "- ■ 빨강 = 확진보조 (Spec>=90%)"
        )

    # 이미지 업로드 (다중)
    st.markdown("**Ensemble(DenseNet + EfficientNet) + TTA + Grad-CAM** 고정 설정으로 분석합니다. "
                "출력 확률은 Per-disease Platt Scaling이 적용된 calibrated probability입니다.")

    uploaded_files = st.file_uploader(
        "흉부 X-ray 이미지 업로드 (PNG / JPEG, 여러 장 가능)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="demo_upload",
    )

    if uploaded_files:
        # 업로드된 이미지 썸네일 미리보기
        st.markdown(f"**업로드된 이미지: {len(uploaded_files)}장**")
        preview_cols = st.columns(min(len(uploaded_files), 6))
        for i, f in enumerate(uploaded_files[:6]):
            with preview_cols[i]:
                img = Image.open(io.BytesIO(f.read()))
                st.image(img, caption=f.name, width=100)
                f.seek(0)
        if len(uploaded_files) > 6:
            st.caption(f"... 외 {len(uploaded_files) - 6}장")

        # 분석 시작 버튼
        if st.button("분석 시작", type="primary", use_container_width=True):
            for idx, uploaded in enumerate(uploaded_files):
                image_bytes = uploaded.read()
                st.markdown(f"### 이미지 {idx + 1}: {uploaded.name}")

                # 1단계: 추론만 (gradcam=false)
                with st.status(f"[{idx+1}/{len(uploaded_files)}] 추론 중...", expanded=True) as status_ui:
                    st.write("📤 추론 요청 전송...")
                    result = call_predict_api(image_bytes, uploaded.name, gradcam=False)
                    if result:
                        st.write(f"✅ 추론 완료 ({result['inference_time_ms']}ms)")
                        status_ui.update(label=f"추론 완료 ({result['inference_time_ms']}ms)", state="complete", expanded=False)
                    else:
                        status_ui.update(label="추론 실패", state="error")
                        continue

                # 추론 결과 즉시 표시
                render_single_result(image_bytes, result)

                # 2단계: Grad-CAM 요청 (백그라운드)
                with st.spinner("Grad-CAM 생성 중..."):
                    gradcam_result = call_predict_api(image_bytes, uploaded.name, gradcam=True, gradcam_top1_only=True)

                if gradcam_result and gradcam_result.get("gradcam_base64"):
                    gradcam_dict = gradcam_result["gradcam_base64"]
                    gradcam_ms = gradcam_result.get("gradcam_time_ms", 0)
                    if gradcam_dict:
                        st.markdown(f"**Grad-CAM 결과** ({gradcam_ms}ms)")
                        cam_cols = st.columns(min(len(gradcam_dict), 4))
                        for j, (disease_name, b64) in enumerate(gradcam_dict.items()):
                            if b64:
                                with cam_cols[j % len(cam_cols)]:
                                    cam_img = Image.open(io.BytesIO(base64.b64decode(b64)))
                                    display = DISEASE_LABELS_KO.get(disease_name, disease_name)
                                    st.image(cam_img, caption=f"Grad-CAM: {display}", use_container_width=True)

                    # Grad-CAM 로그
                    render_processing_log(gradcam_result.get("log", []))

                st.markdown("---")


if __name__ == "__main__":
    main()

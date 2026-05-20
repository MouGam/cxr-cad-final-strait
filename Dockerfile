FROM python:3.12-slim

WORKDIR /workspace

# 시스템 의존성 (OpenCV headless용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0 torchvision==0.21.0 \
    && pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt

# 소스 코드 + 워밍업용 샘플 이미지
COPY app/ /workspace/app/
COPY model_assets/ /workspace/model_assets/
COPY start.sh /workspace/
COPY sample_data/sample_xray.png /workspace/sample_xray.png

# start.sh 실행 권한
RUN chmod +x /workspace/start.sh

# 모델 가중치 (best fold .pth + .onnx + threshold + platt params)
# HF gated repo 인증 문제 회피: 로컬에서 직접 복사
COPY models/densenet121/fold0.pth /workspace/models/densenet121/fold0.pth
COPY models/densenet121/fold0.onnx /workspace/models/densenet121/fold0.onnx
COPY models/densenet121/fold0.onnx.data /workspace/models/densenet121/fold0.onnx.data
COPY models/densenet121/thresholds.json /workspace/models/densenet121/thresholds.json
COPY models/densenet121/platt_params.json /workspace/models/densenet121/platt_params.json
COPY models/densenet121/screening_thresholds.json /workspace/models/densenet121/screening_thresholds.json
COPY models/densenet121/confirmatory_thresholds.json /workspace/models/densenet121/confirmatory_thresholds.json

COPY models/efficientnet_b4/fold3.pth /workspace/models/efficientnet_b4/fold3.pth
COPY models/efficientnet_b4/fold3.onnx /workspace/models/efficientnet_b4/fold3.onnx
COPY models/efficientnet_b4/fold3.onnx.data /workspace/models/efficientnet_b4/fold3.onnx.data
COPY models/efficientnet_b4/thresholds.json /workspace/models/efficientnet_b4/thresholds.json
COPY models/efficientnet_b4/platt_params.json /workspace/models/efficientnet_b4/platt_params.json
COPY models/efficientnet_b4/screening_thresholds.json /workspace/models/efficientnet_b4/screening_thresholds.json
COPY models/efficientnet_b4/confirmatory_thresholds.json /workspace/models/efficientnet_b4/confirmatory_thresholds.json

EXPOSE 8000 8501

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/workspace/start.sh"]

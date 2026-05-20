#!/bin/bash
set -e

export PYTHONPATH=/workspace:${PYTHONPATH}

echo "=== CXR-CAD Serving System ==="
echo "FastAPI: http://localhost:8000"
echo "Streamlit: http://localhost:8501"
echo "Swagger: http://localhost:8000/docs"
echo "================================"

# FastAPI를 백그라운드에서 실행
uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1 &
FASTAPI_PID=$!

# FastAPI 준비 대기 (최대 30초)
echo "FastAPI 서버 시작 대기 중..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "FastAPI 준비 완료"
        break
    fi
    sleep 1
done

# 워밍업: 샘플 이미지로 추론 1회 (ONNX 세션 + 전처리 파이프라인 캐싱)
echo "추론 파이프라인 워밍업 중..."
curl -s -X POST http://localhost:8000/predict \
    -F "file=@/workspace/sample_xray.png" \
    -F "model=ensemble" -F "fold=best" -F "tta=false" -F "gradcam=false" \
    > /dev/null 2>&1 && echo "워밍업 완료" || echo "워밍업 실패 (무시)"

# Streamlit 실행 (포그라운드)
streamlit run app/dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none

# Streamlit 종료 시 FastAPI도 종료
kill $FASTAPI_PID 2>/dev/null || true
